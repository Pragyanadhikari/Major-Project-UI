import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:url_launcher/url_launcher.dart';

class NewsDetailScreen extends StatefulWidget {
  final String newsImage,
      newsTitle,
      newsDate,
      newsLink,
      author,
      description,
      source;
  const NewsDetailScreen({
    super.key,
    required this.newsImage,
    required this.newsTitle,
    required this.newsDate,
    required this.newsLink,
    required this.author,
    required this.description,
    required this.source,
  });

  @override
  State<NewsDetailScreen> createState() => _NewsDetailScreenState();
}

class _NewsDetailScreenState extends State<NewsDetailScreen> {
  final format = DateFormat('MMMM dd, yyyy');
  @override
  Widget build(BuildContext context) {
    final height = MediaQuery.sizeOf(context).height;
    DateTime dateTime = DateTime.parse(widget.newsDate);

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 3,
      ),
      body: Stack(
        children: [
          Container(
            child: Container(
              height: height * .45,
              child: ClipRRect(
                borderRadius: BorderRadius.only(
                    topLeft: Radius.circular(30),
                    topRight: Radius.circular(30)),
                child: CachedNetworkImage(
                  imageUrl: widget.newsImage,
                  fit: BoxFit.cover,
                  placeholder: (context, url) =>
                      Center(child: CircularProgressIndicator()),
                ),
              ),
            ),
          ),
          Container(
            height: height * 0.6,
            margin: EdgeInsets.only(top: height * .4),
            padding: EdgeInsets.only(top: 20, right: 20, left: 20),
            decoration: BoxDecoration(
              color: Colors.grey,
              borderRadius: BorderRadius.only(
                  topLeft: Radius.circular(20), topRight: Radius.circular(20)),
            ),
            child: ListView(
              children: [
                Text(
                  widget.newsTitle,
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    fontSize: 30,
                  ),
                ),
                SizedBox(
                  height: height * 0.01,
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Source:' + widget.author,
                      style: TextStyle(
                        fontWeight: FontWeight.w500,
                        fontSize: 15,
                      ),
                    ),
                    Text(
                      format.format(dateTime),
                      style: TextStyle(
                        fontWeight: FontWeight.w500,
                        fontSize: 15,
                      ),
                    ),
                  ],
                ),
                SizedBox(
                  height: height * 0.01,
                ),
                Text(
                  widget.description,
                  style: TextStyle(
                    fontWeight: FontWeight.w500,
                    fontSize: 20,
                  ),
                ),
                SizedBox(
                  height: 10,
                ),
                Text(
                  "To Know More about, click the link:",
                  style: TextStyle(
                    fontWeight: FontWeight.w500,
                    fontSize: 20,
                  ),
                ),
                GestureDetector(
                  onTap: () async {
                    final url = Uri.parse(widget.newsLink);
                    if (await canLaunchUrl(url)) {
                      await launchUrl(url);
                    } else {
                      throw 'Could not launch $url';
                    }
                  },
                  child: Text(
                    widget.newsLink,
                    style: TextStyle(
                      color: Colors.blue,
                      fontSize: 20, // Optionally change text color
                      decoration:
                          TextDecoration.underline, // Optionally underline
                    ),
                  ),
                )
              ],
            ),
          ),
        ],
      ),
    );
  }
}
