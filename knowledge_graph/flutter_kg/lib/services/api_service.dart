import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // برای اجرا در مرورگر (Chrome)
  static const String baseUrl = 'http://localhost:8000';

  // ۱. ارسال سوال به ایجنت Graph-RAG
  Future<Map<String, dynamic>> askAgent(String question, String targetEntity) async {
    final url = Uri.parse('$baseUrl/ask');
    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'question': question,
          'target_entity': targetEntity,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(utf8.decode(response.bodyBytes));
      } else {
        throw Exception('خطا از سمت سرور: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('عدم اتصال به سرور پایتون: $e');
    }
  }

  // ۲. دریافت تحلیل اثرات بحران از Neo4j
  Future<Map<String, dynamic>> getImpactAnalysis(String sourceEntity) async {
    final url = Uri.parse('$baseUrl/impact-analysis');
    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'source_entity': sourceEntity,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(utf8.decode(response.bodyBytes));
      } else {
        throw Exception('خطا در تحلیل اثرات: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('عدم اتصال به سرور پایتون: $e');
    }
  }
}