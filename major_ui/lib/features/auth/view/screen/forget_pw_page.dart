import 'package:firebase_auth/firebase_auth.dart';
// import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:major_ui/features/auth/view/widget/customTextField.dart';

class ForgetPasswordPage extends StatefulWidget {
  const ForgetPasswordPage({super.key});

  @override
  State<ForgetPasswordPage> createState() => _ForgetPasswordPageState();
}

class _ForgetPasswordPageState extends State<ForgetPasswordPage> {
  final emailController = TextEditingController();

  @override
  void dispose() {
    emailController.dispose();
    super.dispose();
  }

  Future passwordReset() async {
    try {
      await FirebaseAuth.instance
          .sendPasswordResetEmail(email: emailController.text.trim());
      showDialog(
        context: context,
        builder: (context) {
          return AlertDialog(
            content:
                Text('Password Reset link sent to email! Check your email.'),
          );
        },
      );
    } on FirebaseAuthException catch (e) {
      // print(e);
      showDialog(
        context: context,
        builder: (context) {
          return AlertDialog(
            content: Text(e.message.toString()),
          );
        },
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.grey,
        elevation: 3,
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 25.0),
            child: Text(
              'Enter your email:',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 20),
            ),
          ),
          SizedBox(
            height: 10,
          ),
          MyTextField(
            controller: emailController,
            hintText: 'Email',
            obscureText: false,
          ),
          SizedBox(
            height: 10,
          ),
          MaterialButton(
            onPressed: passwordReset,
            child: Text('Reset Password'),
            color: Colors.grey.shade200,
          ),
        ],
      ),
    );
  }
}
