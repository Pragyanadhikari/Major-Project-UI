import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:major_ui/features/auth/view/screen/auth_page.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();

    Timer(
        const Duration(
          seconds: 3,
        ), () {
      Navigator.pushReplacement(
          context, MaterialPageRoute(builder: (context) => AuthPage()));
    });
  }

  @override
  Widget build(BuildContext context) {
    final height = MediaQuery.sizeOf(context).height * 1;
    // final width = MediaQuery.sizeOf(context).width * 1;

    return Scaffold(
      body: Container(
        color: Colors.grey.shade700,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.asset(
              'lib/images/portfolio.jpg',
              fit: BoxFit.cover,
              height: height * 0.5,
            ),
            SizedBox(
              height: height * 0.04,
            ),
            Text(
              'PORTFOLIO MANAGEMENT',
              style: GoogleFonts.daysOne(
                  letterSpacing: .6, color: Colors.grey[100]),
            ),
            SizedBox(
              height: height * 0.04,
            ),
            SpinKitChasingDots(
              color: Colors.white,
              size: 40,
            )
          ],
        ),
      ),
    );
  }
}
