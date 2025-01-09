import 'package:flutter/material.dart';
import 'package:major_ui/features/auth/view/screen/login.dart';
import 'package:major_ui/features/auth/view/screen/register_pages.dart';

class LoginOrRegisterPage extends StatefulWidget {
  const LoginOrRegisterPage({super.key});

  @override
  State<LoginOrRegisterPage> createState() => _LoginOrRegisterPageState();
}

class _LoginOrRegisterPageState extends State<LoginOrRegisterPage> {
  //initailly show login page
  bool showLoginPage = true;

  //toggle between login and register
  void togglePages() {
    setState(() {
      showLoginPage = !showLoginPage;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (showLoginPage) {
      return LoginPage(
          // onTap: togglePages,
          );
    } else {
      return RegisterPages(
        onTap: togglePages,
      );
    }
  }
}
