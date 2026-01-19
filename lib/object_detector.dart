import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class ObjectDetector extends StatefulWidget {
  const ObjectDetector({super.key});

  @override
  State<ObjectDetector> createState() => _ObjectDetectorState();
}

class _ObjectDetectorState extends State<ObjectDetector> with WidgetsBindingObserver {
  CameraController? _cameraController;
  Interpreter? _interpreter;
  FlutterTts? _tts;
  List<String> _labels = [];
  bool _isProcessing = false;
  String _currentLabel = 'Initializing...';
  double _currentScore = 0.0;
  DateTime _lastRun = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _lastSpokenAt = DateTime.fromMillisecondsSinceEpoch(0);
  String _lastSpoken = '';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initAll();
  }

  Future<void> _initAll() async {
    await _initTts();
    await _loadLabels();
    await _loadModel();
    await _initCamera();
  }

  Future<void> _initTts() async {
    final tts = FlutterTts();
    await tts.setLanguage('en-US');
    await tts.setSpeechRate(0.5);
    await tts.setVolume(1.0);
    await tts.setPitch(1.0);
    _tts = tts;
  }

  Future<void> _loadLabels() async {
    final raw = await rootBundle.loadString('assets/bhaudata.txt.txt');
    _labels = raw
        .split('\n')
        .map((line) => line.trim())
        .where((line) => line.isNotEmpty)
        .map((line) {
          final parts = line.split(':');
          return parts.length > 1 ? parts.sublist(1).join(':').trim() : line;
        })
        .toList();
  }

  Future<void> _loadModel() async {
    _interpreter = await Interpreter.fromAsset('assets/yolov8n-cls_float32.tflite');
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    final back = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );
    final controller = CameraController(
      back,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await controller.initialize();
    await controller.startImageStream(_processCameraImage);
    if (mounted) {
      setState(() {
        _cameraController = controller;
      });
    }
  }

  void _processCameraImage(CameraImage image) async {
    if (_interpreter == null || _isProcessing) {
      return;
    }
    final now = DateTime.now();
    if (now.difference(_lastRun) < const Duration(milliseconds: 400)) {
      return;
    }
    _lastRun = now;
    _isProcessing = true;
    try {
      final inputShape = _interpreter!.getInputTensor(0).shape;
      final inputHeight = inputShape[1];
      final inputWidth = inputShape[2];

      final rgbImage = _convertYuv420ToImage(image);
      final resized = img.copyResize(
        rgbImage,
        width: inputWidth,
        height: inputHeight,
      );

      final inputBuffer = Float32List(inputWidth * inputHeight * 3);
      var index = 0;
      for (var y = 0; y < inputHeight; y++) {
        for (var x = 0; x < inputWidth; x++) {
          final pixel = resized.getPixel(x, y);
          inputBuffer[index++] = img.getRed(pixel) / 255.0;
          inputBuffer[index++] = img.getGreen(pixel) / 255.0;
          inputBuffer[index++] = img.getBlue(pixel) / 255.0;
        }
      }

      final outputShape = _interpreter!.getOutputTensor(0).shape;
      if (outputShape.length == 2 && outputShape[0] == 1) {
        final output = List.filled(outputShape[0] * outputShape[1], 0.0)
            .reshape(outputShape);
        _interpreter!.run(inputBuffer.reshape([1, inputHeight, inputWidth, 3]), output);
        final scores = output[0] as List<double>;
        final top = _argMax(scores);
        final label = top < _labels.length ? _labels[top] : 'Unknown';
        final score = scores[top];
        _updateResult(label, score);
      } else {
        _updateResult('Unsupported model output', 0.0);
      }
    } catch (e) {
      _updateResult('Error: $e', 0.0);
    } finally {
      _isProcessing = false;
    }
  }

  void _updateResult(String label, double score) {
    if (!mounted) {
      return;
    }
    setState(() {
      _currentLabel = label;
      _currentScore = score;
    });
    final shouldSpeak = score >= 0.4 &&
        label != _lastSpoken &&
        DateTime.now().difference(_lastSpokenAt) > const Duration(seconds: 2);
    if (shouldSpeak) {
      _lastSpoken = label;
      _lastSpokenAt = DateTime.now();
      _tts?.speak(label.replaceAll('_', ' '));
    }
  }

  int _argMax(List<double> values) {
    var bestIndex = 0;
    var bestScore = -double.infinity;
    for (var i = 0; i < values.length; i++) {
      if (values[i] > bestScore) {
        bestScore = values[i];
        bestIndex = i;
      }
    }
    return bestIndex;
  }

  img.Image _convertYuv420ToImage(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel ?? 1;
    final imgImage = img.Image(width, height);
    for (var y = 0; y < height; y++) {
      final uvRow = y ~/ 2;
      for (var x = 0; x < width; x++) {
        final uvCol = x ~/ 2;
        final uvIndex = uvRow * uvRowStride + uvCol * uvPixelStride;
        final yIndex = y * image.planes[0].bytesPerRow + x;
        final yp = image.planes[0].bytes[yIndex];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];
        var r = (yp + 1.402 * (vp - 128)).round();
        var g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128)).round();
        var b = (yp + 1.772 * (up - 128)).round();
        r = max(0, min(255, r));
        g = max(0, min(255, g));
        b = max(0, min(255, b));
        imgImage.setPixelRgba(x, y, r, g, b);
      }
    }
    return imgImage;
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _cameraController;
    if (controller == null) {
      return;
    }
    if (state == AppLifecycleState.inactive) {
      controller.stopImageStream();
      controller.dispose();
      _cameraController = null;
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _interpreter?.close();
    _tts?.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Object Detection'),
      ),
      body: controller == null || !controller.value.isInitialized
          ? const Center(child: CircularProgressIndicator())
          : Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(controller),
                Align(
                  alignment: Alignment.bottomCenter,
                  child: Container(
                    width: double.infinity,
                    color: Colors.black.withOpacity(0.6),
                    padding: const EdgeInsets.all(12),
                    child: Text(
                      '$_currentLabel (${(_currentScore * 100).toStringAsFixed(1)}%)',
                      style: const TextStyle(color: Colors.white, fontSize: 18),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
              ],
            ),
    );
  }
}
