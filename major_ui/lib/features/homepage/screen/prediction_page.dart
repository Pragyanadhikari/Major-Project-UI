// import 'package:fl_chart/fl_chart.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:major_ui/features/auth/view/screen/login.dart';
import 'package:major_ui/features/auth/view/widget/app_bar.dart';
import 'package:major_ui/features/auth/view/widget/navigation_bar.dart';
import 'package:major_ui/features/homepage/screen/calculator_screen.dart';
import 'package:major_ui/features/homepage/screen/news_screen.dart';
import 'package:major_ui/features/homepage/screen/portfolio_page.dart';
// import 'package:major_ui/services/api_service.dart';

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  _PredictionPageState createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final user = FirebaseAuth.instance.currentUser!;
  int _selectedIndex = 0;
  String _prediction = 'Prediction will appear here';
  // List<double> _predictions = [];

  // Handles tab navigation
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

  // Fetches stock prediction and updates the state
  // Future<void> _getStockPrediction() async {
  //   try {
  //     final result = await ApiService().getPrediction('NULB');
  //     setState(() {
  //       _prediction = result['prediction'].toString();
  //       _predictions = result['historicalPredictions'] != null
  //           ? List<double>.from(result['historicalPredictions'])
  //           : [];
  //     });
  //   } catch (e) {
  //     setState(() {
  //       _prediction = 'Error fetching prediction';
  //       _predictions = [];
  //     });
  //   }
  // }

  // Generates chart data points
  // List<FlSpot> _generateChartData() {
  //   return List.generate(
  //     _predictions.length,
  //     (index) => FlSpot(index.toDouble(), _predictions[index]),
  //   );
  // }

  // Customizes bottom titles for the graph
  // AxisTitles _bottomTitles() {
  //   return AxisTitles(
  //     sideTitles: SideTitles(
  //       showTitles: true,
  //       interval: 1,
  //       getTitlesWidget: (value, meta) {
  //         String label = value.toInt() < _predictions.length
  //             ? 'Day ${value.toInt() + 1}'
  //             : '';
  //         return Text(label, style: const TextStyle(fontSize: 12));
  //       },
  //     ),
  //   );
  // }

  // Builds the line chart for predictions
  // Widget _buildGraph() {
  //   if (_predictions.isEmpty) {
  //     return const Text(
  //       'No prediction data available',
  //       style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
  //     );
  //   }

  //   return SizedBox(
  //     height: 300,
  //     child: LineChart(
  //       LineChartData(
  //         gridData: FlGridData(show: true),
  //         titlesData: FlTitlesData(
  //           bottomTitles: _bottomTitles(),
  //           leftTitles: AxisTitles(
  //             sideTitles: SideTitles(
  //               showTitles: true,
  //               reservedSize: 40,
  //               getTitlesWidget: (value, meta) {
  //                 return Text(
  //                   value.toStringAsFixed(1),
  //                   style: const TextStyle(fontSize: 12),
  //                 );
  //               },
  //             ),
  //           ),
  //         ),
  //         borderData: FlBorderData(
  //           show: true,
  //           border: Border.all(color: Colors.grey, width: 1),
  //         ),
  //         lineBarsData: [
  //           LineChartBarData(
  //             spots: _generateChartData(),
  //             isCurved: true,
  //             color: Colors.blue,
  //             belowBarData: BarAreaData(
  //               show: true,
  //               color: Colors.blue,
  //             ),
  //           ),
  //         ],
  //       ),
  //     ),
  //   );
  // }

  // Signs the user out and redirects to the login page
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

  // Displays an error dialog
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

  // Build the UI of the PredictionPage
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
            // Disclaimer Box
            _buildDisclaimerBox(width),
            const SizedBox(height: 20),
            // Get Prediction Button
            _buildPredictionButton(),
            const SizedBox(height: 20),
            // Prediction Text
            _buildPredictionText(),
            const SizedBox(height: 30),
            // Prediction Graph
            // _buildGraph(),
          ],
        ),
      ),
      bottomNavigationBar: CustomBottomNavBar(
        currentIndex: _selectedIndex,
        onTabChange: _onTabChange,
      ),
    );
  }

  // Builds the disclaimer box widget
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

  // Builds the prediction button widget
  Widget _buildPredictionButton() {
    return Center(
      child: ElevatedButton(
        // onPressed: _getStockPrediction,
        onPressed: () {},
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 16.0),
          textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        child: const Text('Get Prediction'),
      ),
    );
  }

  // Builds the prediction text widget
  Widget _buildPredictionText() {
    return Text(
      _prediction,
      style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
    );
  }
}
