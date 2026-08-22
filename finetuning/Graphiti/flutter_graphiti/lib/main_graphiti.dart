import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(const GraphitiApp());

class GraphitiApp extends StatelessWidget {
  const GraphitiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.orange, useMaterial3: true),
      home: const AgentMemoryScreen(),
    );
  }
}

class AgentMemoryScreen extends StatefulWidget {
  const AgentMemoryScreen({super.key});

  @override
  State<AgentMemoryScreen> createState() => _AgentMemoryScreenState();
}

class _AgentMemoryScreenState extends State<AgentMemoryScreen> {
  String _subject = "User";
  String _memoryResult = "برای مشاهده حافظه فعال ایجنت کلیک کنید.";
  bool _loading = false;

  Future<void> _fetchMemory() async {
    setState(() => _loading = true);
    try {
      final res = await http.post(
        Uri.parse('http://127.0.0.1:8006/get_active_memory'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'subject': _subject}),
      );
      final data = jsonDecode(utf8.decode(res.bodyBytes));
      List activeFacts = data['active_facts'] ?? [];

      setState(() {
        _memoryResult = "حافظه فعال فعلی کاربر:\n" +
            (activeFacts.isEmpty ? "هیچ فکت فعالی یافت نشد." : activeFacts.join("\n")) +
            "\n\n(تعداد کل اپیزودهای ثبت‌شده در تاریخچه: ${data['total_episodes_in_history']})";
      });
    } catch (e) {
      setState(() => _memoryResult = "خطا در اتصال به Graphiti: $e");
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('حافظه زنده ایجنت (Graphiti)')),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            TextField(
              decoration: const InputDecoration(
                labelText: 'سوژه حافظه (مثلاً: User)',
                border: OutlineInputBorder(),
              ),
              onChanged: (v) => _subject = v,
            ),
            const SizedBox(height: 15),
            ElevatedButton(
              onPressed: _loading ? null : _fetchMemory,
              child: _loading ? const CircularProgressIndicator() : const Text('بازیابی فکت‌های معتبر زمان'),
            ),
            const SizedBox(height: 25),
            Card(
              elevation: 3,
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: Text(_memoryResult, style: const TextStyle(fontSize: 16)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}