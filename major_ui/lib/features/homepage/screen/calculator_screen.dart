import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:intl/intl.dart';
import 'package:major_ui/features/auth/view/screen/login.dart';
import 'package:major_ui/features/auth/view/widget/app_bar.dart';
import 'package:major_ui/features/auth/view/widget/navigation_bar.dart';
import 'package:major_ui/features/homepage/screen/news_screen.dart';
import 'package:major_ui/features/homepage/screen/portfolio_page.dart';
import 'package:major_ui/features/homepage/screen/prediction_page.dart';

class CalculatorScreen extends StatefulWidget {
  const CalculatorScreen({super.key});

  @override
  State<CalculatorScreen> createState() => _CalculatorScreenState();
}

class _CalculatorScreenState extends State<CalculatorScreen> {
  final user = FirebaseAuth.instance.currentUser;
  int _selectedIndex = 3;

  final TextEditingController _openController = TextEditingController();
  final TextEditingController _closeController = TextEditingController();
  final TextEditingController _quantityController = TextEditingController();
  String? _result;
  bool _isCalculating = false;

  double _selectedTaxRate = 5.0;

  double? totalAmount;
  double? brokerCommission;
  double? sebonCharge;
  double? dpCharge;
  double? receivableAmount;
  double? capitalGain;
  double? tax;
  double? profitOrLoss;

  @override
  void dispose() {
    _openController.dispose();
    _closeController.dispose();
    _quantityController.dispose();
    super.dispose();
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
    if (index == _selectedIndex) return;

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
        break;
    }
  }

  void _calculateProfitLoss() {
    if (_openController.text.isEmpty ||
        _closeController.text.isEmpty ||
        _quantityController.text.isEmpty) {
      setState(() {
        _result =
            "Please enter valid Base Price, Selling Price, and Quantity values.";
      });
      return;
    }

    final double? basePrice = double.tryParse(_openController.text);
    final double? sellingPrice = double.tryParse(_closeController.text);
    final int? quantity = int.tryParse(_quantityController.text);

    if (basePrice == null ||
        sellingPrice == null ||
        quantity == null ||
        quantity <= 0) {
      setState(() {
        _result =
            "Please ensure all inputs are valid and quantity is greater than 0.";
      });
      return;
    }

    setState(() {
      _isCalculating = true;
    });

    Future.delayed(const Duration(milliseconds: 300), () {
      setState(() {
        totalAmount = sellingPrice * quantity;

        // Broker commission calculation
        if (totalAmount! <= 2500) {
          brokerCommission = 10;
        } else if (totalAmount! <= 50000) {
          brokerCommission = totalAmount! * 0.0036;
        } else if (totalAmount! <= 500000) {
          brokerCommission = totalAmount! * 0.0033;
        } else if (totalAmount! <= 1000000) {
          brokerCommission = totalAmount! * 0.0031;
        } else if (totalAmount! <= 10000000) {
          brokerCommission = totalAmount! * 0.0033;
        } else {
          brokerCommission = totalAmount! * 0.0024;
        }

        // Other charges
        sebonCharge = totalAmount! * 0.00015;
        dpCharge = 25.0;

        receivableAmount =
            totalAmount! - brokerCommission! - sebonCharge! - dpCharge!;

        // Capital gain and tax
        capitalGain = receivableAmount! - (basePrice * quantity);
        tax = capitalGain! * (_selectedTaxRate / 100);

        profitOrLoss = capitalGain! - tax!;

        final NumberFormat formatter = NumberFormat("#,##0.00");

        _result =
            "Share Amount: Rs ${formatter.format(totalAmount)}\n\nBroker Commission: Rs ${formatter.format(brokerCommission)}\n\nSEBON Charge: Rs ${formatter.format(sebonCharge)}\n\nDP Charge: Rs ${formatter.format(dpCharge)}\n\nReceivable Amount: Rs ${formatter.format(receivableAmount)}\n\nCapital Gain: Rs ${formatter.format(capitalGain)}\n\nTax ($_selectedTaxRate%): Rs ${formatter.format(tax)}\n\n${profitOrLoss! >= 0 ? "Profit: Rs ${formatter.format(profitOrLoss)}" : "Loss: Rs ${formatter.format(-profitOrLoss!)}"}";

        _isCalculating = false;
      });
    });
  }

  void _showTaxInfoDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text("Tax Information"),
          content: const Text(
              "Select 7.5% if you are selling it within 12 months of buying, else select 5%."),
          actions: <Widget>[
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: const Text('Close'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final height = MediaQuery.sizeOf(context).height;
    final width = MediaQuery.sizeOf(context).width;

    return Scaffold(
      appBar: CustomAppBar(
        signUserOut: () => signUserOut(context),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.start,
                children: [
                  SizedBox(height: height * 0.02),

                  // Title
                  Center(
                    child: Container(
                      alignment: Alignment.center,
                      height: height * 0.05,
                      width: width * 0.9,
                      decoration: BoxDecoration(
                        color: Colors.grey.shade500,
                        border:
                            Border.all(color: Colors.grey.shade900, width: 2),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Text(
                        'Profit/Loss Calculator',
                        style: TextStyle(
                          fontSize: 20,
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(height: height * 0.02),

                  Container(
                    width: width * 0.92,
                    padding: const EdgeInsets.symmetric(vertical: 10.0),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(10),
                      color: Colors.grey.shade400,
                    ),
                    child: Column(
                      children: [
                        _buildTextField(
                          controller: _openController,
                          label: 'Base Price',
                        ),
                        SizedBox(height: height * 0.02),
                        _buildTextField(
                          controller: _closeController,
                          label: 'Selling Price',
                        ),
                        SizedBox(height: height * 0.02),
                        _buildTextField(
                          controller: _quantityController,
                          label: 'Number of Stocks:',
                        ),
                        SizedBox(height: height * 0.02),
                        Row(
                          children: [
                            const Text(
                              "Select Tax Percentage:",
                              style: TextStyle(fontSize: 16),
                            ),
                            const SizedBox(width: 30),
                            DropdownButton<double>(
                              value: _selectedTaxRate,
                              items: const [
                                DropdownMenuItem(
                                  value: 5.0,
                                  child: Text("5%"),
                                ),
                                DropdownMenuItem(
                                  value: 7.5,
                                  child: Text("7.5%"),
                                ),
                              ],
                              onChanged: (value) {
                                setState(() {
                                  _selectedTaxRate = value!;
                                });
                              },
                            ),
                            SizedBox(width: 30),
                            IconButton(
                              icon: const Icon(
                                Icons.help_outline,
                                color: Colors.blue,
                              ),
                              onPressed: _showTaxInfoDialog,
                            ),
                          ],
                        ),
                        SizedBox(height: height * 0.02),
                        ElevatedButton(
                          onPressed:
                              _isCalculating ? null : _calculateProfitLoss,
                          child: const Text('Calculate'),
                        ),
                      ],
                    ),
                  ),

                  SizedBox(height: height * 0.02),

                  // Result Section (Below Calculate Button)
                  _result != null
                      ? Container(
                          alignment: Alignment.topLeft,
                          width: width * 0.9,
                          padding: const EdgeInsets.all(16.0),
                          decoration: BoxDecoration(
                            color: Colors.grey[500],
                            border: Border.all(
                                color: Colors.grey.shade400, width: 2),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              _buildResultRow('Share Amount:', totalAmount),
                              const SizedBox(
                                height: 5,
                              ),
                              _buildResultRow(
                                  'Broker Commission:', brokerCommission),
                              const SizedBox(
                                height: 5,
                              ),
                              _buildResultRow('SEBON Charge:', sebonCharge),
                              const SizedBox(
                                height: 5,
                              ),
                              _buildResultRow('DP Charge:', dpCharge),
                              const SizedBox(
                                height: 5,
                              ),
                              _buildResultRow(
                                  'Receivable Amount:', receivableAmount),
                              const SizedBox(
                                height: 5,
                              ),
                              _buildResultRow('Capital Gain:', capitalGain),
                              const SizedBox(
                                height: 5,
                              ),
                              _buildResultRow('Tax ($_selectedTaxRate%):', tax),
                              const SizedBox(
                                height: 5,
                              ),
                              _buildResultRow(
                                  profitOrLoss! >= 0 ? 'Profit:' : 'Loss:',
                                  profitOrLoss),
                            ],
                          ),
                        )
                      : const SizedBox(),
                ],
              ),
            ),
          ],
        ),
      ),
      bottomNavigationBar: CustomBottomNavBar(
        currentIndex: _selectedIndex,
        onTabChange: _onTabChange,
      ),
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String label,
  }) {
    return TextField(
      controller: controller,
      keyboardType: TextInputType.number,
      decoration: InputDecoration(
        labelText: label,
        border: OutlineInputBorder(),
      ),
    );
  }

  Widget _buildResultRow(String label, double? value) {
    final NumberFormat formatter = NumberFormat("#,##0.00");
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: const TextStyle(fontSize: 16)),
        Text(
          'Rs ${value != null ? formatter.format(value) : '0.00'}',
          style: const TextStyle(fontSize: 16),
        ),
      ],
    );
  }
}
