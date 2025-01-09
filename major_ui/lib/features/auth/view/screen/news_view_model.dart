import 'package:major_ui/features/homepage/repository/news_repository.dart';
import 'package:major_ui/news_content.dart';

class NewsViewModel {
  final _repo = NewsRepository();

  Future<NewsContent> fetchNewsChannelHeadlineApi() async {
    final response = await _repo.fetchNewsChannelHeadlineApi();
    return response;
  }
}
