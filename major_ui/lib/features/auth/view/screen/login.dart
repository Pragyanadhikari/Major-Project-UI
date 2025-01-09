import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:major_ui/features/auth/view/screen/forget_pw_page.dart';
import 'package:major_ui/features/auth/view/screen/register_pages.dart';
import 'package:major_ui/features/auth/view/widget/customTextField.dart';
import 'package:major_ui/features/auth/view/widget/my_button.dart';
import 'package:major_ui/features/auth/view/widget/square_tile.dart';
import 'package:major_ui/features/homepage/screen/prediction_page.dart';
import 'package:major_ui/services/auth_service.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  //sign user in
  Future signUserIn() async {
    //show loading circle
    showDialog(
      context: context,
      builder: (context) {
        return const Center(
          child: CircularProgressIndicator(),
        );
      },
    );

    try {
      final userCredential =
          await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: _emailController.text.trim(),
        password: _passwordController.text.trim(),
      );

      // User is signed in
      print("Signed in as: ${userCredential.user?.email}");

      // Pop the loading dialog
      Navigator.pop(context);

      // Navigate to the home page after successful login
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
            builder: (context) =>
                PredictionPage()), // Replace with your actual HomePage widget
      );
    } on FirebaseAuthException catch (e) {
      Navigator.pop(context); // Pop the loading circle if error occurs
      showError(e.message ?? "An error occurred");
    }
  }

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  // Show error messages in dialog
  void showError(String message) {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          backgroundColor: Colors.deepPurple,
          title: Center(
            child: Text(
              message,
              style: const TextStyle(color: Colors.white),
            ),
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[350],
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            scrollDirection: Axis.vertical,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(
                  height: 50,
                ),
                const Icon(
                  Icons.lock,
                  size: 100,
                ),
                const SizedBox(
                  height: 50,
                ),
                Text(
                  'Welcome User.',
                  style: TextStyle(color: Colors.grey[700], fontSize: 16),
                ),
                const SizedBox(
                  height: 25,
                ),
                MyTextField(
                  controller: _emailController,
                  hintText: 'Email',
                  obscureText: false,
                ),
                const SizedBox(
                  height: 10,
                ),
                MyTextField(
                  controller: _passwordController,
                  hintText: 'Password',
                  obscureText: true,
                ),
                const SizedBox(
                  height: 10,
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 25.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      GestureDetector(
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) {
                                return ForgetPasswordPage();
                              },
                            ),
                          );
                        },
                        child: Text(
                          'Forget Password?',
                          style: TextStyle(
                              color: Colors.blue, fontWeight: FontWeight.bold),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(
                  height: 25,
                ),
                MyButton(
                  text: "Sign In",
                  onTap: signUserIn,
                ),
                const SizedBox(
                  height: 50,
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 25.0),
                  child: Row(
                    children: [
                      Expanded(
                        child: Divider(
                          thickness: 0.5,
                          color: Colors.grey[400],
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 10.0),
                        child: Text(
                          'Or continue with',
                          style: TextStyle(color: Colors.grey[700]),
                        ),
                      ),
                      Expanded(
                        child: Divider(
                          thickness: 0.5,
                          color: Colors.grey[400],
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(
                  height: 50,
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    SquareTile(
                      onTap: () async {
                        UserCredential? userCredential =
                            await AuthService().signInWithGoogle(context);
                        if (userCredential != null) {
                          // Handle successful Google sign-in here (if needed)
                        } else {
                          // Handle Google sign-in failure here
                          print("Google sign-in failed");
                        }
                      },
                      imagePath: 'lib/images/google.png',
                    ),
                    const SizedBox(
                      width: 10,
                    ),
                    // SquareTile(imagePath: 'lib/images/apple.png') // Optional for Apple sign-in
                  ],
                ),
                const SizedBox(
                  height: 50,
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      'Not a member?',
                      style: TextStyle(color: Colors.grey[700]),
                    ),
                    const SizedBox(
                      width: 4,
                    ),
                    GestureDetector(
                      onTap: () {
                        // Navigate to Register Page
                        Navigator.pushReplacement(
                            context,
                            MaterialPageRoute(
                                builder: (context) => RegisterPages(
                                      onTap: () {},
                                    )
                                    )
                                    );
                      },
                      child: const Text(
                        'Register Now.',
                        style: TextStyle(
                            color: Colors.blue, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ],
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
}
