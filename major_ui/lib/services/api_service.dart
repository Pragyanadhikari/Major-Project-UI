import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // Your Flask API URL
  static const String _baseUrl = 'http://your_flask_server_ip:5000/predict';

  // Function to send the CSV file path and get prediction from Flask
  Future<Map<String, dynamic>> getPrediction(String csvPath) async {
    try {
      // Prepare data to send in the request body
      final Map<String, dynamic> requestData = {
        'input_data': csvPath, // Send the path to the CSV file
      };

      // Send a POST request to the Flask API
      final response = await http.post(
        Uri.parse(_baseUrl),
        headers: {'Content-Type': 'application/json'},
        body: json.encode(requestData),
      );

      if (response.statusCode == 200) {
        // Parse and return the response if the request is successful
        final Map<String, dynamic> responseData = json.decode(response.body);
        return responseData;  // Contains the prediction result
      } else {
        throw Exception('Failed to load prediction');
      }
    } catch (e) {
      print('Error: $e');
      throw Exception('Failed to load prediction');
    }
  }
}
