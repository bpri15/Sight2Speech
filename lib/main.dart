import 'package:flutter/material.dart';
import 'object_detector.dart'; // Import the ObjectDetector widget

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Object Detection',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: ObjectDetector(), // Use the ObjectDetector widget as the home
    );
  }
}
