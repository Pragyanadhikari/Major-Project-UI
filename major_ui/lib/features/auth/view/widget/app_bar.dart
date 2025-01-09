import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class CustomAppBar extends StatelessWidget implements PreferredSizeWidget {
  final VoidCallback signUserOut; // Accepts a callback for sign out

  const CustomAppBar({Key? key, required this.signUserOut}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return AppBar(
      backgroundColor: Colors.grey.shade900,
      title: Text(
        'STOCK PREDICTION AND PORTFOLIO MANAGEMENT',
        overflow: TextOverflow.clip,
        softWrap: true,
        style: GoogleFonts.pangolin(
          fontSize: 17,
          fontWeight: FontWeight.bold,
          color: Colors.grey.shade100,
        ),
        textAlign: TextAlign.center,
      ),
      actions: [
        IconButton(
          onPressed: signUserOut, // Call the function when pressed
          icon: const Icon(Icons.logout),
          color: Colors.white,
          iconSize: 30,
        ),
      ],
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}
