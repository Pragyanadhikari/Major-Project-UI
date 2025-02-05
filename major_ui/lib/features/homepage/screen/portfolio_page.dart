import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:csv/csv.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:major_ui/features/auth/view/screen/login.dart';
import 'package:major_ui/features/auth/view/widget/app_bar.dart';
import 'package:major_ui/features/auth/view/widget/navigation_bar.dart';
import 'package:major_ui/features/homepage/screen/calculator_screen.dart';
import 'package:major_ui/features/homepage/screen/news_screen.dart';
import 'package:major_ui/features/homepage/screen/prediction_page.dart';
import 'package:major_ui/features/services/firestore.dart';

class PortfolioPage extends StatefulWidget {
  const PortfolioPage({super.key});

  @override
  State<PortfolioPage> createState() => _PortfolioPageState();
}

class _PortfolioPageState extends State<PortfolioPage> {
  final FirestoreService firestoreService = FirestoreService();
  final FirebaseAuth auth = FirebaseAuth.instance;
  String? _selectedCompany;
  int? _numberOfStocks;
  final TextEditingController _stockController = TextEditingController();
  List<String> _companyNames = [];
  int _selectedIndex = 1;

  @override
  void initState() {
    super.initState();
    _loadCompanyNames();
  }

  Future<void> _loadCompanyNames() async {
    try {
      final String csvData =
          await rootBundle.loadString('assets/companies.csv');
      final List<List<dynamic>> rows =
          const CsvToListConverter().convert(csvData);

      setState(() {
        _companyNames = rows.skip(1).map((row) => row[1].toString()).toList();
      });
    } catch (e) {
      debugPrint("Error loading company names from CSV: $e");
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

  void openAddBox(String? docId) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            DropdownButtonFormField<String>(
              value: _selectedCompany,
              hint: const Text('Select a Company'),
              items: _companyNames.map((String company) {
                return DropdownMenuItem<String>(
                  value: company,
                  child: Text(company),
                );
              }).toList(),
              onChanged: (String? newValue) {
                setState(() {
                  _selectedCompany = newValue;
                });
              },
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _stockController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                labelText: 'Number of Stocks',
                border: OutlineInputBorder(),
              ),
              onChanged: (value) {
                _numberOfStocks = int.tryParse(value);
              },
            ),
          ],
        ),
        actions: [
          ElevatedButton(
            onPressed: () {
              if (_selectedCompany != null && _numberOfStocks != null) {
                final userId = auth.currentUser?.uid;
                if (userId != null) {
                  if (docId == null) {
                    firestoreService.addStock(
                        _selectedCompany!, _numberOfStocks!, userId);
                  } else {
                    firestoreService.updateStock(
                        docId, _selectedCompany!, _numberOfStocks!);
                  }
                  setState(() {
                    _selectedCompany = null;
                    _stockController.clear();
                  });
                  Navigator.pop(context);
                } else {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('User not logged in')),
                  );
                }
              } else {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Please fill in all fields')),
                );
              }
            },
            child: const Text('Add'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppBar(
        signUserOut: () => signUserOut(context),
      ),
      body: StreamBuilder<QuerySnapshot>(
        stream: firestoreService.getStockStream(),
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return Center(
              child: Text("Error loading stocks: ${snapshot.error}"),
            );
          }

          if (snapshot.hasData) {
            List stocksList = snapshot.data!.docs;

            return ListView.builder(
              itemCount: stocksList.length,
              itemBuilder: (context, index) {
                DocumentSnapshot document = stocksList[index];
                String docId = document.id;
                Map<String, dynamic> data =
                    document.data() as Map<String, dynamic>;

                String company = data['company'];
                int numberOfStocks = data['number_of_stocks'];

                return ListTile(
                  title: Text(company),
                  subtitle: Text('Number of Stocks: $numberOfStocks'),
                  trailing: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      IconButton(
                        onPressed: () {
                          showDialog(
                            context: context,
                            builder: (context) => AlertDialog(
                              title: const Text('Confirm Deletion'),
                              content: Text(
                                  'Do you really want to delete this stock?\n\nCompany: $company\nNo. of Stocks: $numberOfStocks'),
                              actions: [
                                TextButton(
                                  onPressed: () => Navigator.of(context).pop(),
                                  child: const Text('Cancel'),
                                ),
                                TextButton(
                                  onPressed: () {
                                    firestoreService.deleteStock(docId);
                                    Navigator.of(context).pop();
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(
                                          content:
                                              Text('Stock deleted: $company')),
                                    );
                                  },
                                  child: const Text('Delete',
                                      style: TextStyle(color: Colors.red)),
                                ),
                              ],
                            ),
                          );
                        },
                        icon: const Icon(Icons.delete),
                      ),
                      IconButton(
                        onPressed: () {
                          // Open an input dialog first
                          showDialog(
                            context: context,
                            builder: (context) {
                              String? newCompany = _selectedCompany ?? company;
                              TextEditingController newStockController =
                                  TextEditingController(
                                      text: numberOfStocks.toString());

                              return AlertDialog(
                                title: const Text('Enter New Stock Details'),
                                content: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    DropdownButtonFormField<String>(
                                      value: newCompany,
                                      hint: const Text('Select New Company'),
                                      items:
                                          _companyNames.map((String company) {
                                        return DropdownMenuItem<String>(
                                          value: company,
                                          child: Text(company),
                                        );
                                      }).toList(),
                                      onChanged: (String? newValue) {
                                        newCompany = newValue;
                                      },
                                    ),
                                    const SizedBox(height: 16),
                                    TextField(
                                      controller: newStockController,
                                      keyboardType: TextInputType.number,
                                      decoration: const InputDecoration(
                                        labelText: 'New Number of Stocks',
                                        border: OutlineInputBorder(),
                                      ),
                                    ),
                                  ],
                                ),
                                actions: [
                                  TextButton(
                                    onPressed: () =>
                                        Navigator.of(context).pop(),
                                    child: const Text('Cancel'),
                                  ),
                                  TextButton(
                                    onPressed: () {
                                      if (newCompany != null &&
                                          newStockController.text.isNotEmpty) {
                                        int? newStockCount = int.tryParse(
                                            newStockController.text);

                                        if (newStockCount != null) {
                                          Navigator.of(context)
                                              .pop(); // Close input dialog

                                          // Show confirmation dialog with previous and new values
                                          showDialog(
                                            context: context,
                                            builder: (context) => AlertDialog(
                                              title:
                                                  const Text('Confirm Update'),
                                              content: Text(
                                                'Do you want to update the stock as follows?\n\n'
                                                'From: $company ($numberOfStocks stocks)\n'
                                                'To: $newCompany ($newStockCount stocks)',
                                              ),
                                              actions: [
                                                TextButton(
                                                  onPressed: () =>
                                                      Navigator.of(context)
                                                          .pop(),
                                                  child: const Text('Cancel'),
                                                ),
                                                TextButton(
                                                  onPressed: () {
                                                    firestoreService
                                                        .updateStock(
                                                            docId,
                                                            newCompany!,
                                                            newStockCount);
                                                    Navigator.of(context)
                                                        .pop(); // Close confirmation dialog
                                                    ScaffoldMessenger.of(
                                                            context)
                                                        .showSnackBar(
                                                      const SnackBar(
                                                          content: Text(
                                                              'Stock updated successfully')),
                                                    );
                                                  },
                                                  child: const Text('Update',
                                                      style: TextStyle(
                                                          color: Colors.green)),
                                                ),
                                              ],
                                            ),
                                          );
                                        }
                                      }
                                    },
                                    child: const Text('Next'),
                                  ),
                                ],
                              );
                            },
                          );
                        },
                        icon: const Icon(Icons.update),
                      ),
                    ],
                  ),
                );
              },
            );
          } else {
            return const Center(
              child: CircularProgressIndicator(),
            );
          }
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => openAddBox(null),
        child: const Icon(Icons.add),
      ),
      bottomNavigationBar: CustomBottomNavBar(
        currentIndex: _selectedIndex,
        onTabChange: _onTabChange,
      ),
    );
  }
}
