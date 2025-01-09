import 'package:flutter/material.dart';
import 'package:google_nav_bar/google_nav_bar.dart';

class CustomBottomNavBar extends StatelessWidget {
  final int currentIndex;
  final Function(int) onTabChange;

  const CustomBottomNavBar({
    Key? key,
    required this.currentIndex,
    required this.onTabChange,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color.fromARGB(255, 156, 156, 156),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 15.0, vertical: 20),
        child: GNav(
          gap: 5,
          backgroundColor: const Color.fromARGB(255, 156, 156, 156),
          color: Colors.white,
          activeColor: Colors.white,
          tabBackgroundColor: Colors.black87,
          padding: const EdgeInsets.all(18),
          selectedIndex: currentIndex,
          onTabChange: onTabChange,
          tabs: const [
            GButton(
              icon: Icons.auto_graph,
              text: 'Prediction',
            ),
            GButton(
              icon: Icons.business_center_rounded,
              text: 'Portfolio',
            ),
            GButton(
              icon: Icons.newspaper,
              text: 'News',
            ),
            GButton(
              icon: Icons.calculate,
              text: 'Calculator',
            ),
          ],
        ),
      ),
    );
  }
}
