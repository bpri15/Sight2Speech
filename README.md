# Sight2Speech

Real-time, on-device object recognition with spoken feedback for accessibility use cases. The app runs fully offline using a YOLOv8 classification model in TensorFlow Lite. On lower-end phones, inference latency is noticeable due to limited CPU/GPU resources.

## Why this project
- Make object recognition usable hands-free with spoken output.
- Keep data private by running all inference on-device.
- Provide a simple, low-latency loop from camera to speech.

## Key features
- Live camera stream with on-device inference.
- YOLOv8 classification (top-1 label + confidence).
- Text-to-speech announcements with confidence gating and cooldown.
- Minimal UI overlay for the current prediction.

## Tech stack
- Flutter + Dart
- camera
- tflite_flutter
- flutter_tts
- image
- TensorFlow Lite (YOLOv8 classification)

## How it works (high level)
1. Capture camera frames in YUV420 format.
2. Convert to RGB and resize to model input size.
3. Run YOLOv8 classification with TFLite.
4. Pick top-1 label and display + speak it.
5. Throttle inference to keep the UI responsive.

## Architecture
- Camera stream -> YUV420 to RGB -> Resize -> TFLite Interpreter -> Top-1 label
- Output -> UI overlay + TTS

## Performance notes
- Inference runs roughly every 400ms to reduce CPU load.
- On lower-end phones, latency is noticeable because the processor is limited.

## Setup
Prerequisites:
- Flutter SDK
- Android Studio or Xcode for device builds

Steps:
```bash
flutter pub get
```

## Run
```bash
flutter run
```

## Assets
- Model: `assets/yolov8n-cls_float32.tflite`
- Labels: `assets/bhaudata.txt.txt`

## Project layout
- `lib/main.dart`: App entry point
- `lib/object_detector.dart`: Camera + inference + TTS pipeline

## Limitations
- Classification only (no bounding boxes).
- Accuracy depends on the model and labels file.
- Latency varies by device performance.

## Ideas for improvement
- Switch to a detection model for bounding boxes.
- Add FPS/latency overlay and profiling.
- Add a low-power mode or adjustable inference interval.
