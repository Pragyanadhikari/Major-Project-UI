import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:major_ui/features/auth/view/screen/login_or_register_page.dart';
import 'package:major_ui/features/homepage/screen/prediction_page.dart';

// import 'home_page.dart';

class AuthPage extends StatefulWidget {
  const AuthPage({super.key});

  @override
  State<AuthPage> createState() => _AuthPageState();
}

class _AuthPageState extends State<AuthPage> {
  bool showLoginPage=true;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: StreamBuilder<User?>(
        stream: FirebaseAuth.instance.authStateChanges(),
        builder: (context, snapshot) {
          //user is logged in
          if (snapshot.hasData) {
            return PredictionPage();
          }

          //user is not logged in
          else {
            return LoginOrRegisterPage();
          }
        },
      ),
    );
  }

  dynamic HomePage() => HomePage();
}
