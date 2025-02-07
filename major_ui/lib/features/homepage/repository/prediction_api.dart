import 'dart:convert';
import 'package:http/http.dart' as http;

class PredictionApi {
  static Future<Map<String, dynamic>> getPrediction(String company) async {
    final String apiUrl = "http://192.168.1.96:1234/predict$company";

    try {
      final response = await http.get(Uri.parse(apiUrl));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return {
          'prediction': 'Prediction for $company: ${data['action']}',
          'image': data['img'], // Base64 image string from API
        };
      } else {
        throw "Error: ${response.reasonPhrase}";
      }
    } catch (e) {
      throw "Failed to fetch prediction: $e";
    }
  }
}
