import 'package:flutter/material.dart';
import 'image_classifier_page.dart'; // 👈 امپورت فایل صفحه اصلی  

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ResNet Classifier',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const ImageClassifierPage(), // 👈 صدا زدن صفحه اصلی
    );
  }
}