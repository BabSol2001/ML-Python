import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(const PyGApp());

class PyGApp extends StatelessWidget {
  const PyGApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const ToolRecommendationScreen(),
    );
  }
}

class ToolRecommendationScreen extends StatefulWidget {
  const ToolRecommendationScreen({super.key});

  @override
  State<ToolRecommendationScreen> createState() => _ToolRecommendationScreenState();
}

class _ToolRecommendationScreenState extends State<ToolRecommendationScreen> {
  int taskId = 0;
  int toolId = 3;
  String result = "برای بررسی پیشنهاد ابزار کلیک کنید.";
  bool loading = false;

  Future<void> checkRecommendation() async {
    setState(() => loading = true);
    try {
      final res = await http.post(
        Uri.parse('http://127.0.0.1:8000/predict_tool'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'task_id': taskId, 'tool_id': toolId}),
      );
      final data = jsonDecode(res.body);

      if (res.statusCode == 200) {
        setState(() {
          result = "امتیاز تناسب: ${data['match_score']}%\n"
                   "نتیجه: ${data['recommended'] ? 'پیشنهاد می‌شود 🎯' : 'مناسب نیست ❌'}";
        });
      } else {
        setState(() => result = "خطا: ${data['detail']}");
      }
    } catch (e) {
      setState(() => result = "خطا در اتصال به API: $e");
    } finally {
      setState(() => loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('پیشنهاد هوشمند ابزار (PyG)')),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            TextField(
              decoration: const InputDecoration(labelText: 'شناسه مسئله (Task ID 0-2)', border: OutlineInputBorder()),
              keyboardType: TextInputType.number,
              onChanged: (v) => taskId = int.tryParse(v) ?? 0,
            ),
            const SizedBox(height: 15),
            TextField(
              decoration: const InputDecoration(labelText: 'شناسه ابزار (Tool ID 3-4)', border: OutlineInputBorder()),
              keyboardType: TextInputType.number,
              onChanged: (v) => toolId = int.tryParse(v) ?? 3,
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: loading ? null : checkRecommendation,
              child: loading ? const CircularProgressIndicator() : const Text('ارزیابی تناسب'),
            ),
            const SizedBox(height: 30),
            Text(result, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold), textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }
}