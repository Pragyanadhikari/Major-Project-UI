import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:flutter/material.dart';
import 'package:major_ui/features/homepage/screen/prediction_page.dart';

class AuthService {
  // Google Sign-In
  Future<UserCredential?> signInWithGoogle(BuildContext context) async {
    try {
      // Begin interactive sign-in process
      final GoogleSignInAccount? gUser = await GoogleSignIn().signIn();

      // Check if the user is null (if the user cancels the sign-in)
      if (gUser == null) {
        // The user canceled the sign-in
        print("User canceled sign-in");
        return null;
      }

      // Obtain auth details from the Google Sign-In request
      final GoogleSignInAuthentication gAuth = await gUser.authentication;

      // Create a new credential for the user
      final credential = GoogleAuthProvider.credential(
        accessToken: gAuth.accessToken,
        idToken: gAuth.idToken,
      );

      // Finally sign in using Firebase authentication
      UserCredential userCredential =
          await FirebaseAuth.instance.signInWithCredential(credential);

      // Navigate to the home page (or another page) after successful sign-in
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
            builder: (context) =>
                PredictionPage()), // Replace with your target page
      );

      return userCredential;
    } catch (e) {
      // Handle errors here
      print("Error during Google Sign-In: $e");

      // Show an alert to the user in case of error
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: Text('Sign-In Error'),
          content: Text('An error occurred during sign-in: $e'),
          actions: <Widget>[
            TextButton(
              child: Text('OK'),
              onPressed: () => Navigator.of(context).pop(),
            ),
          ],
        ),
      );
      return null;
    }
  }
}
