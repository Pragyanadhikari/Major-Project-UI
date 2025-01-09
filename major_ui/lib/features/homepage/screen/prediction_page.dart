import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:major_ui/features/auth/view/screen/login.dart';
import 'package:major_ui/features/auth/view/widget/app_bar.dart';
import 'package:major_ui/features/auth/view/widget/navigation_bar.dart';
import 'package:major_ui/features/homepage/screen/calculator_screen.dart';
import 'package:major_ui/features/homepage/screen/news_screen.dart';
import 'package:major_ui/features/homepage/screen/portfolio_page.dart';
import 'package:major_ui/services/api_service.dart'; // Import the service to fetch prediction

class PredictionPage extends StatefulWidget {
  PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final user = FirebaseAuth.instance.currentUser!;
  int _selectedIndex = 0;
  String _prediction = ''; // Store the prediction result

  // You no longer need a dropdown for company selection.

  void _onTabChange(int index) {
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
        Navigator.pushReplacement(
            context, MaterialPageRoute(builder: (_) => NewsScreen()));
        break;
      case 3:
        Navigator.pushReplacement(
            context, MaterialPageRoute(builder: (_) => CalculatorScreen()));
        break;
    }
  }

  Future<void> _getStockPrediction() async {
    try {
      // Directly call prediction for NULB
      final result = await ApiService().getPrediction('NULB');
      setState(() {
        _prediction = result['prediction'].toString(); // Set prediction
      });
    } catch (e) {
      setState(() {
        _prediction = 'Error fetching prediction'; // Handle errors
      });
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
    final height = MediaQuery.sizeOf(context).height;
    final width = MediaQuery.sizeOf(context).width;

    return Scaffold(
      appBar: CustomAppBar(
        signUserOut: () => signUserOut(context),
      ),
      body: Column(
        children: [
          Container(
            height: 10,
            color: Colors.white,
          ),
          Container(
            height: height * 0.25,
            width: width,
            decoration: BoxDecoration(
              color: Colors.yellow.shade200,
              borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(30), topRight: Radius.circular(30)),
            ),
            child: const Text(
              'Disclaimer: This prediction is just for educational purpose and does not actually suggest anyone to follow it blindly. The model for prediction is always in learning phase. The prediction may not always be correct.',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20),
            ),
          ),

          // Remove the dropdown since you're predicting for NULB only

          // Button to get prediction
          ElevatedButton(
            onPressed: _getStockPrediction,
            child: const Text('Get Prediction'),
          ),

          // Display the prediction result
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(
              _prediction.isNotEmpty ? 'Prediction: $_prediction' : '',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
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
