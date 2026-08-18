

import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GNN Link Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const GNNPredictorScreen(),
    );
  }
}

class GNNPredictorScreen extends StatefulWidget {
  const GNNPredictorScreen({super.key});

  @override
  State<GNNPredictorScreen> createState() => _GNNPredictorScreenState();
}

class _GNNPredictorScreenState extends State<GNNPredictorScreen> {
  int sourceNode = 0;
  int targetNode = 2;
  String result = "اطلاعات گره‌ها را وارد کرده و روی دکمه کلیک کنید.";
  bool isLoading = false;
  bool isError = false;

  Future<void> predictLink() async {
    setState(() {
      isLoading = true;
      isError = false;
    });

    // آدرس پیش‌فرض برای وب، ویندوز و مک
    String baseUrl = 'http://127.0.0.1:8000'; 
  
    // بررسی هوشمند امولاتور اندروید بدون نیاز به dart:io
    if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
      baseUrl = 'http://10.0.2.2:8000';
    }

    final url = Uri.parse('$baseUrl/predict_link');

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'source_node': sourceNode,
          'target_node': targetNode,
        }),
      );

      final data = jsonDecode(utf8.decode(response.bodyBytes));

      if (response.statusCode == 200) {
        setState(() {
          double probability = (data['link_probability'] as num).toDouble() * 100;
          bool exists = data['exists'] as bool;

          result = "احتمال وجود رابطه: ${probability.toStringAsFixed(1)}%\n\n"
                   "نتیجه: ${exists ? 'رابطه برقرار است ✅' : 'رابطه‌ای وجود ندارد ❌'}";
        });
      } else if (response.statusCode == 400) {
        // دریافت خطای اعتبارسنجی ID از سمت FastAPI
        setState(() {
          isError = true;
          result = "خطای ورودی: ${data['detail']}";
        });
      } else {
        setState(() {
          isError = true;
          result = "خطای ناشناخته از سمت سرور (کد: ${response.statusCode})";
        });
      }
    } catch (e) {
      setState(() {
        isError = true;
        result = "خطا در برقراری ارتباط با سرور FastAPI\n($e)";
      });
    } finally {
      setState(() => isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('پیش‌بینی روابط گراف (GNN)'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'شناسه گره‌ها را بین 0 تا 4 وارد کنید:',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            TextFormField(
              initialValue: '0',
              decoration: const InputDecoration(
                labelText: 'گره مبدا (Source Node ID)',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.hub_outlined),
              ),
              keyboardType: TextInputType.number,
              onChanged: (val) => sourceNode = int.tryParse(val) ?? 0,
            ),
            const SizedBox(height: 16),
            TextFormField(
              initialValue: '2',
              decoration: const InputDecoration(
                labelText: 'گره مقصد (Target Node ID)',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.ads_click_outlined),
              ),
              keyboardType: TextInputType.number,
              onChanged: (val) => targetNode = int.tryParse(val) ?? 2,
            ),
            const SizedBox(height: 24),
            ElevatedButton(
              onPressed: isLoading ? null : predictLink,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              child: isLoading
                  ? const SizedBox(
                      height: 24,
                      width: 24,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text(
                      'پیش‌بینی احتمال رابطه',
                      style: TextStyle(fontSize: 16),
                    ),
            ),
            const SizedBox(height: 32),
            Card(
              elevation: 2,
              color: isError ? Colors.red.shade50 : Colors.deepPurple.shade50,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
                side: BorderSide(
                  color: isError ? Colors.red.shade200 : Colors.deepPurple.shade200,
                ),
              ),
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: Text(
                  result,
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: isError ? Colors.red.shade900 : Colors.deepPurple.shade900,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}