import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:major_ui/features/auth/view/screen/login.dart';
import 'package:major_ui/features/auth/view/widget/app_bar.dart';
import 'package:major_ui/features/auth/view/widget/navigation_bar.dart';
import 'package:major_ui/features/homepage/screen/calculator_screen.dart';
import 'package:major_ui/features/homepage/screen/news_screen.dart';
import 'package:major_ui/features/homepage/screen/portfolio_page.dart';
import 'package:flutter/services.dart'; // For loading local assets
import 'dart:typed_data';
import '../repository/prediction_api.dart';

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  _PredictionPageState createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final user = FirebaseAuth.instance.currentUser!;
  int _selectedIndex = 0;
  Map<String, dynamic> _prediction = {};
  Map<String, dynamic> _tftprediction = {};
  String? _selectedCompany;
  Uint8List? imageBytes;

  void _onTabChange(int index) {
    final pages = [
      const PredictionPage(),
      const PortfolioPage(),
      const NewsScreen(),
      const CalculatorScreen(),
    ];

    if (index != _selectedIndex) {
      setState(() {
        _selectedIndex = index;
      });
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => pages[index]),
      );
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
      _showErrorDialog(context, 'Failed to sign out. Please try again.');
    }
  }

  Future<void> _fetchPrediction() async {
    if (_selectedCompany == null) {
      _showErrorDialog(context, "Please select a company first.");
      return;
    }

    // Print when the button is pressed
    print("Fetching prediction for company: $_selectedCompany");

    try {
      Map<String, dynamic> prediction =
          await PredictionApi.getPrediction(_selectedCompany!);
      setState(() {
        _prediction = prediction;

        imageBytes = base64Decode(prediction['image'] ?? '');
      });

      print("Prediction received: $_prediction"); // Print prediction result
    } catch (e) {
      _showErrorDialog(context, "Failed to fetch prediction: $e");
    }
  }

  void _showErrorDialog(BuildContext context, String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.of(context).size.width;

    return Scaffold(
      appBar: CustomAppBar(
        signUserOut: () => signUserOut(context),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildDisclaimerBox(width),
            const SizedBox(height: 20),
            _buildCompanyDropdown(),
            const SizedBox(height: 20),
            _buildPredictionButton(),
            const SizedBox(height: 20),
            _buildRLPredictionText(),
            const SizedBox(height: 30),
            _buildImage(), // Display the image
            const SizedBox(height: 30),
            _buildTFTPredictionText(),
          ],
        ),
      ),
      bottomNavigationBar: CustomBottomNavBar(
        currentIndex: _selectedIndex,
        onTabChange: _onTabChange,
      ),
    );
  }

  Widget _buildDisclaimerBox(double width) {
    return Container(
      width: width,
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: Colors.yellow.shade200,
        borderRadius: BorderRadius.circular(15),
      ),
      child: const Text(
        'Disclaimer: This prediction is for educational purposes. The model is constantly learning, and predictions may not always be accurate.',
        style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
        textAlign: TextAlign.center,
      ),
    );
  }

  Widget _buildCompanyDropdown() {
    return DropdownButtonFormField<String>(
      value: _selectedCompany,
      decoration: const InputDecoration(
        icon: Icon(Icons.search),
        labelText: "Select Company",
        border: OutlineInputBorder(),
      ),
      items: ["NUBL", "LLBS"].map((company) {
        return DropdownMenuItem(
          value: company,
          child: Text(company),
        );
      }).toList(),
      onChanged: (value) {
        setState(() {
          _selectedCompany = value;
        });
      },
    );
  }

  Widget _buildPredictionButton() {
    return Center(
      child: ElevatedButton(
        onPressed: _fetchPrediction,
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 16.0),
          textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        child: const Text('Get Prediction'),
      ),
    );
  }

  Widget _buildRLPredictionText() {
    String predictionText =
        _prediction['prediction'] ?? 'No prediction available';

    return Center(
      child: Column(
        children: [
          Text(
            predictionText,
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildTFTPredictionText() {
    String predictionText =
        _prediction['tft_prediction'] ?? 'No prediction available';

    return Center(
      child: Column(
        children: [
          Text(
            predictionText,
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildImage() {
    if (imageBytes != null && imageBytes!.isNotEmpty) {
      return Image.memory(imageBytes!);
    } else {
      return const Center(
          child: Text(
        "Select Company to get prediction result.",
        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
      ));
    }
  }
}
