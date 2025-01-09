import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:major_ui/news_content.dart';

class NewsRepository {
  Future<NewsContent> fetchNewsChannelHeadlineApi() async {
    String url =
        'https://newsdata.io/api/1/latest?apikey=pub_62316307f4a18149cad0acb6414d582a85893&domain=onlinekhabar';
    final response = await http.get(Uri.parse(url));

    if (response.statusCode == 200) {
      String decodeText = utf8.decode(response.bodyBytes);

      final jsonData = jsonDecode(decodeText);
      return NewsContent.fromJson(jsonData);
    } else {
      throw Exception('Error');
    }
  }
}
