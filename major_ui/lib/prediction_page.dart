import 'package:flutter/material.dart';

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  _PredictionPageState createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  int _selectedIndex = 0;

  static const List<Widget> _pages = <Widget>[
    Center(
      child: Text(
        'Prediction Page',
        style: TextStyle(fontSize: 24),
      ),
    ),
    Center(
      child: Text(
        'Portfolio Management Page',
        style: TextStyle(fontSize: 24),
      ),
    ),
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "STOCK PREDICTION AND PORTFOLIO MANAGEMENT",
          style: TextStyle(
            color: Colors.white,
            fontSize: 16,
          ),
        ),
        backgroundColor: Colors.blueAccent,
        centerTitle: true,
      ),
      body: _pages.elementAt(_selectedIndex),
      bottomNavigationBar: BottomNavigationBar(
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.bar_chart),
            label: 'Prediction',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.account_balance_wallet),
            label: 'Portfolio Management',
          ),
        ],
        currentIndex: _selectedIndex,
        selectedItemColor: Colors.blueAccent,
        onTap: _onItemTapped,
      ),
    );
  }
}
