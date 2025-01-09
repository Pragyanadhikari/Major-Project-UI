import 'package:cached_network_image/cached_network_image.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:major_ui/features/auth/view/screen/login.dart';
import 'package:major_ui/features/auth/view/screen/news_view_model.dart';
import 'package:major_ui/features/auth/view/widget/app_bar.dart';
import 'package:major_ui/features/auth/view/widget/navigation_bar.dart';
import 'package:major_ui/features/homepage/screen/calculator_screen.dart';
import 'package:major_ui/features/homepage/screen/news_detail_screen.dart';
import 'package:major_ui/features/homepage/screen/portfolio_page.dart';
import 'package:major_ui/features/homepage/screen/prediction_page.dart';
import 'package:major_ui/news_content.dart';

class NewsScreen extends StatefulWidget {
  const NewsScreen({super.key});

  @override
  State<NewsScreen> createState() => _NewsScreenState();
}

class _NewsScreenState extends State<NewsScreen> {
  final user = FirebaseAuth.instance.currentUser!;
  int _selectedIndex = 2; // Default index for NewsScreen

  void _onTabChange(int index) {
    if (index == _selectedIndex) return; // Prevent redundant navigation

    setState(() {
      _selectedIndex = index;
    });

    switch (index) {
      case 0:
        Navigator.pushReplacement(
            context, MaterialPageRoute(builder: (_) => PredictionPage()));
        break;
      case 1:
        Navigator.pushReplacement(
            context, MaterialPageRoute(builder: (_) => PortfolioPage()));
        break;
      case 2:
        // Do nothing since already on NewsScreen
        break;
      case 3:
        Navigator.pushReplacement(
            context, MaterialPageRoute(builder: (_) => CalculatorScreen()));
        break;
    }
  }
  Future<void> signUserOut(BuildContext context) async {
    try {
      await FirebaseAuth.instance.signOut();
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const LoginPage()),
      );
    } catch (e) {
      debugPrint("Sign out failed: $e");
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Error'),
          content: const Text('Failed to sign out. Please try again.'),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('OK'),
            ),
          ],
        ),
      );
    }
  }
  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width;
    final height = MediaQuery.sizeOf(context).height;
    final format = DateFormat('MMMM dd, yyyy');
    NewsViewModel newsviewmodel = NewsViewModel();

    return Scaffold(
      appBar: CustomAppBar(
        signUserOut: () => signUserOut(context),
      ),
      body: ListView(
        padding: const EdgeInsets.symmetric(vertical: 10),
        children: [
          SizedBox(
            height: height * 0.75,
            width: width * 0.9,
            child: FutureBuilder<NewsContent>(
              future: newsviewmodel.fetchNewsChannelHeadlineApi(),
              builder: (BuildContext context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(
                    child: SpinKitFadingCircle(
                      size: 50,
                      color: Colors.grey,
                    ),
                  );
                } else if (snapshot.hasError) {
                  return Center(
                    child: Text(
                      'Error: ${snapshot.error}',
                      style: const TextStyle(color: Colors.red, fontSize: 16),
                    ),
                  );
                } else {
                  return ListView.builder(
                    itemCount: snapshot.data?.results?.length ?? 0,
                    scrollDirection: Axis.horizontal,
                    itemBuilder: (context, index) {
                      final newsItem = snapshot.data!.results![index];
                      final dateTime = DateTime.parse(newsItem.pubDate!);

                      return InkWell(
                        onTap: () {
                          Navigator.push(
                              context,
                              MaterialPageRoute(
                                  builder: (context) => NewsDetailScreen(
                                        newsImage: newsItem.imageUrl.toString(),
                                        newsTitle: newsItem.title.toString(),
                                        newsDate: newsItem.pubDate.toString(),
                                        newsLink: newsItem.link.toString(),
                                        author: newsItem.sourceId.toString(),
                                        description:
                                            newsItem.description.toString(),
                                        source: newsItem.sourceUrl.toString(),
                                      )));
                        },
                        child: SizedBox(
                          child: Stack(
                            alignment: Alignment.center,
                            children: [
                              Container(
                                height: height * 0.8,
                                width: width * 0.8,
                                padding: EdgeInsets.symmetric(
                                  horizontal: height * 0.02,
                                ),
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(15),
                                  child: CachedNetworkImage(
                                    imageUrl: newsItem.imageUrl!,
                                    fit: BoxFit.cover,
                                    placeholder: (context, url) => spinkit2,
                                    errorWidget: (context, url, error) =>
                                        const Icon(
                                      Icons.error_outline_outlined,
                                      color: Colors.red,
                                    ),
                                  ),
                                ),
                              ),
                              Positioned(
                                bottom: 20,
                                child: Card(
                                  elevation: 5,
                                  color: Colors.grey.shade500,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(16),
                                  ),
                                  child: Container(
                                    padding: const EdgeInsets.all(15),
                                    alignment: Alignment.bottomCenter,
                                    height: height * 0.22,
                                    child: Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      crossAxisAlignment:
                                          CrossAxisAlignment.center,
                                      children: [
                                        SizedBox(
                                          width: width * 0.7,
                                          child: Text(
                                            newsItem.title!,
                                            maxLines: 3,
                                            overflow: TextOverflow.ellipsis,
                                            style:
                                                const TextStyle(fontSize: 17),
                                          ),
                                        ),
                                        const Spacer(),
                                        SizedBox(
                                          width: width * 0.7,
                                          child: Row(
                                            mainAxisAlignment:
                                                MainAxisAlignment.spaceBetween,
                                            children: [
                                              Expanded(
                                                child: Text(
                                                  snapshot.data!.results![index]
                                                      .creator
                                                      .toString(),
                                                  maxLines: 2,
                                                  overflow:
                                                      TextOverflow.ellipsis,
                                                  style: const TextStyle(
                                                      fontSize: 12),
                                                ),
                                              ),
                                              const SizedBox(width: 5),
                                              Text(
                                                format.format(dateTime),
                                                maxLines: 1,
                                                overflow: TextOverflow.ellipsis,
                                                style:
                                                    GoogleFonts.jacquesFrancois(
                                                  fontSize: 10,
                                                ),
                                              ),
                                            ],
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                              const SizedBox(
                                height: 10,
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  );
                }
              },
            ),
          ),
        ],
      ),
      bottomNavigationBar: CustomBottomNavBar(
        currentIndex: _selectedIndex,
        onTabChange: _onTabChange,
      ),
    );
  }
}

const spinkit2 = SpinKitFadingCircle(
  color: Colors.grey,
  size: 40,
);
