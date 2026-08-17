import 'package:flutter/material.dart';
import 'services/api_service.dart';

void main() {
  runApp(const SmartBizApp());
}

class SmartBizApp extends StatelessWidget {
  const SmartBizApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SmartBiz Knowledge Graph',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.indigo,
        fontFamily: 'Vazir', // در صورت وجود فونت فارسی
      ),
      // اعمال راست‌چین برای کل برنامه
      builder: (context, child) {
        return Directionality(
          textDirection: TextDirection.rtl,
          child: child!,
        );
      },
      home: const DashboardScreen(),
    );
  }
}

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final ApiService _apiService = ApiService();
  final TextEditingController _questionController = TextEditingController();
  final TextEditingController _entityController = TextEditingController(text: 'Neo4j'); // مقدار پیش‌فرض واقعی
  
  String _responseAnswer = 'هنوز سوالی پرسیده نشده است.';
  List<dynamic> _affectedEntities = [];
  bool _isLoading = false;

  @override
  void dispose() {
    _questionController.dispose();
    _entityController.dispose();
    super.dispose();
  }

  void _handleAsk() async {
    final question = _questionController.text.trim();
    final entity = _entityController.text.trim();

    if (question.isEmpty) return;

    setState(() => _isLoading = true);
    try {
      final res = await _apiService.askAgent(question, entity);
      setState(() {
        _responseAnswer = res['answer'] ?? 'پاسخی دریافت نشد.';
      });
    } catch (e) {
      setState(() {
        _responseAnswer = 'خطا: $e';
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _handleImpactAnalysis() async {
    final entity = _entityController.text.trim();
    if (entity.isEmpty) return;

    setState(() => _isLoading = true);
    try {
      final res = await _apiService.getImpactAnalysis(entity);
      setState(() {
        _affectedEntities = res['affected_entities'] ?? [];
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('خطا در دریافت اطلاعات Neo4j: $e')),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('داشبورد هوشمند Graph-RAG'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // کارت ورودی‌ها
            Card(
              elevation: 3,
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: Column(
                  children: [
                    TextField(
                      controller: _entityController,
                      decoration: const InputDecoration(
                        labelText: 'نام گره/موجودیت هدف در Neo4j',
                        hintText: 'مثلاً: Neo4j یا Cypher یا Customer',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.hub),
                      ),
                    ),
                    const SizedBox(height: 10),
                    TextField(
                      controller: _questionController,
                      decoration: const InputDecoration(
                        labelText: 'سوال خود را بپرسید...',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.question_answer),
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: _isLoading ? null : _handleAsk,
                            icon: const Icon(Icons.send),
                            label: const Text('پرسش از ایجنت'),
                          ),
                        ),
                        const SizedBox(width: 10),
                        ElevatedButton.icon(
                          onPressed: _isLoading ? null : _handleImpactAnalysis,
                          icon: const Icon(Icons.analytics),
                          label: const Text('تحلیل اثر Neo4j'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.orange.shade100,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 15),
            
            if (_isLoading) const LinearProgressIndicator(),
            const SizedBox(height: 10),

            // نمایش نتایج
            Expanded(
              child: ListView(
                children: [
                  const Text(
                    '🤖 پاسخ ایجنت Graph-RAG:',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  const SizedBox(height: 5),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade100,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.grey.shade300),
                    ),
                    child: Text(_responseAnswer),
                  ),
                  const SizedBox(height: 20),

                  if (_affectedEntities.isNotEmpty) ...[
                    const Text(
                      '🔍 گره‌های متأثر و مرتبط در Neo4j:',
                      style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      runSpacing: 6,
                      children: _affectedEntities.map((entity) {
                        return Chip(
                          avatar: const Icon(Icons.bubble_chart, size: 18),
                          label: Text(entity.toString()),
                          backgroundColor: Colors.indigo.shade50,
                        );
                      }).toList(),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}