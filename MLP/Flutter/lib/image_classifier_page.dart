import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:dio/dio.dart';

class ImageClassifierPage extends StatefulWidget {
  const ImageClassifierPage({super.key});

  @override
  State<ImageClassifierPage> createState() => _ImageClassifierPageState();
}

class _ImageClassifierPageState extends State<ImageClassifierPage> {
  File? _selectedImage;
  bool _isLoading = false;
  String? _predictedClass;
  String? _confidence;
  String? _errorMessage;

  final ImagePicker _picker = ImagePicker();
  final Dio _dio = Dio();

  // ⚠️ آدرس سرور FastAPI:
  // - برای شبیه‌ساز اندروید (Android Emulator): از 10.0.2.2 استفاده کنید.
  // - برای دستگاه واقعی یا iOS Simulator: از IP محلی سیستم (مثلا 192.168.1.50) استفاده کنید.
  final String _apiUrl = 'http://192.168.0.147:8000/predict';

  // ۱. متد انتخاب تصویر از گالری یا دوربین
  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85, // فشرده‌سازی جهت افزایش سرعت ارسال
      );

      if (pickedFile != null) {
        setState(() {
          _selectedImage = File(pickedFile.path);
          _predictedClass = null;
          _confidence = null;
          _errorMessage = null;
        });
        
        // بلافاصله پس از انتخاب، تصویر به FastAPI ارسال می‌شود
        await _uploadAndPredictImage(_selectedImage!);
      }
    } catch (e) {
      setState(() {
        _errorMessage = "خطا در انتخاب تصویر: $e";
      });
    }
  }

  // ۲. متد ارسال تصویر به FastAPI با استفاده از Dio (MultipartForm)
  Future<void> _uploadAndPredictImage(File imageFile) async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      // ساخت فایل Multipart
      String fileName = imageFile.path.split('/').last;
      FormData formData = FormData.fromMap({
        "file": await MultipartFile.fromFile(
          imageFile.path,
          filename: fileName,
        ),
      });

      // ارسال درخواست POST به FastAPI Endpoint
      Response response = await _dio.post(
        _apiUrl,
        data: formData,
        options: Options(
          headers: {
            "Accept": "application/json",
          },
          connectTimeout: const Duration(seconds: 10),
          receiveTimeout: const Duration(seconds: 10),
        ),
      );

      // ۳. پردازش خروجی دریافتی از سرور
      if (response.statusCode == 200) {
        final data = response.data;
        setState(() {
          _predictedClass = data['prediction']['class'];
          _confidence = data['prediction']['confidence'];
        });
      }
    } on DioException catch (e) {
      String msg = "خطا در برقراری ارتباط با سرور";
      if (e.type == DioExceptionType.connectionTimeout) {
        msg = "زمان ارتباط با سرور به پایان رسید (Timeout)";
      } else if (e.response != null) {
        msg = "خطای سرور: ${e.response?.data['detail'] ?? e.response?.statusCode}";
      }
      setState(() {
        _errorMessage = msg;
      });
    } catch (e) {
      setState(() {
        _errorMessage = "خطای ناشناخته: $e";
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('تشخیص تصویر با ResNet-18'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // باکس نمایش تصویر
            Container(
              height: 280,
              decoration: BoxDecoration(
                color: Colors.grey[200],
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.grey[400]!),
              ),
              child: _selectedImage != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(15),
                      child: Image.file(_selectedImage!, fit: BoxFit.cover),
                    )
                  : const Center(
                      child: Text(
                        'هیچ تصویری انتخاب نشده است',
                        style: TextStyle(color: Colors.grey),
                      ),
                    ),
            ),
            const SizedBox(height: 20),

            // دکمه‌های انتخاب تصویر
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isLoading ? null : () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('گالری'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isLoading ? null : () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('دوربین'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 30),

            // لودینگ یا نتیجه پیش‌بینی
            if (_isLoading)
              const Column(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 12),
                  Text('در حال آنالیز تصویر توسط هوش مصنوعی...'),
                ],
              )
            else if (_predictedClass != null)
              Card(
                color: Colors.green[50],
                elevation: 2,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                  side: BorderSide(color: Colors.green[300]!),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      const Text(
                        'کلاس تشخیص داده شده:',
                        style: TextStyle(fontSize: 14, color: Colors.grey),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        _predictedClass!.toUpperCase(),
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.green,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'درصد اطمینان مدل: $_confidence',
                        style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
                      ),
                    ],
                  ),
                ),
              )
            else if (_errorMessage != null)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red[50],
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _errorMessage!,
                  style: const TextStyle(color: Colors.red),
                  textAlign: TextAlign.center,
                ),
              ),
          ],
        ),
      ),
    );
  }
}