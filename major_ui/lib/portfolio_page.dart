import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

class PortfolioPage extends StatefulWidget {
  const PortfolioPage({super.key});

  @override
  _PortfolioPageState createState() => _PortfolioPageState();
}

class _PortfolioPageState extends State<PortfolioPage> {
  List<Map<String, dynamic>> _portfolio = [];

  @override
  void initState() {
    super.initState();
    _loadPortfolio();
  }

  Future<void> _loadPortfolio() async {
    final prefs = await SharedPreferences.getInstance();
    final String? portfolioString = prefs.getString('portfolio');
    if (portfolioString != null) {
      setState(() {
        _portfolio =
            List<Map<String, dynamic>>.from(json.decode(portfolioString));
      });
    }
  }

  Future<void> _savePortfolio() async {
    final prefs = await SharedPreferences.getInstance();
    prefs.setString('portfolio', json.encode(_portfolio));
  }

  void _addStock() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        String stockName = '';
        String numberOfStocks = '';

        return AlertDialog(
          title: const Text('Add Stock'),
          content: SingleChildScrollView(
            child: Column(
              children: [
                TextField(
                  decoration: const InputDecoration(hintText: 'Stock Name'),
                  onChanged: (value) {
                    stockName = value;
                  },
                ),
                TextField(
                  decoration:
                      const InputDecoration(hintText: 'Number of Stocks'),
                  onChanged: (value) {
                    numberOfStocks = value;
                  },
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                setState(() {
                  _portfolio.add({
                    'Stock Name': stockName,
                    'Number of Stocks': numberOfStocks,
                    'LTP': 'N/A',
                    'Close Value': 'N/A',
                    'Open Value': 'N/A',
                    'Suggestion': 'N/A',
                  });
                });
                _savePortfolio();
                Navigator.of(context).pop();
              },
              child: const Text('Add'),
            ),
          ],
        );
      },
    );
  }

  void _editStock(int index) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        String stockName = _portfolio[index]['Stock Name'];
        String numberOfStocks = _portfolio[index]['Number of Stocks'];

        return AlertDialog(
          title: const Text('Edit Stock'),
          content: SingleChildScrollView(
            child: Column(
              children: [
                TextField(
                  decoration: const InputDecoration(hintText: 'Stock Name'),
                  controller: TextEditingController(text: stockName),
                  onChanged: (value) {
                    stockName = value;
                  },
                ),
                TextField(
                  decoration:
                      const InputDecoration(hintText: 'Number of Stocks'),
                  controller: TextEditingController(text: numberOfStocks),
                  onChanged: (value) {
                    numberOfStocks = value;
                  },
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                setState(() {
                  _portfolio[index]['Stock Name'] = stockName;
                  _portfolio[index]['Number of Stocks'] = numberOfStocks;
                });
                _savePortfolio();
                Navigator.of(context).pop();
              },
              child: const Text('Update'),
            ),
            TextButton(
              onPressed: () {
                setState(() {
                  _portfolio.removeAt(index);
                });
                _savePortfolio();
                Navigator.of(context).pop();
              },
              child: const Text('Delete'),
            ),
          ],
        );
      },
    );
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
      body: ListView.builder(
        itemCount: _portfolio.length,
        itemBuilder: (context, index) {
          final stock = _portfolio[index];
          return Card(
            child: ListTile(
              title: Text(stock['Stock Name']!),
              subtitle: Text(
                'Number of Stocks: ${stock['Number of Stocks']}\n'
                'LTP: ${stock['LTP']}\n'
                'Close Value: ${stock['Close Value']}\n'
                'Open Value: ${stock['Open Value']}\n'
                'Suggestion: ${stock['Suggestion']}',
              ),
              trailing: IconButton(
                icon: const Icon(Icons.edit),
                onPressed: () {
                  _editStock(index);
                },
              ),
            ),
          );
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _addStock,
        child: const Icon(Icons.add),
      ),
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
        currentIndex: 1,
        selectedItemColor: Colors.blueAccent,
        onTap: (index) {
          if (index == 0) {
            Navigator.pop(context);
          }
        },
      ),
    );
  }
}
