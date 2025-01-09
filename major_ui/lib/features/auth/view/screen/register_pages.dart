import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:major_ui/features/auth/view/screen/login.dart';
import 'package:major_ui/features/auth/view/widget/customTextField.dart';
import 'package:major_ui/features/auth/view/widget/my_button.dart';
import 'package:major_ui/features/auth/view/widget/square_tile.dart';
import 'package:major_ui/services/auth_service.dart';

class RegisterPages extends StatefulWidget {
  final Function()? onTap;
  const RegisterPages({super.key, required this.onTap});

  @override
  State<RegisterPages> createState() => _RegisterPagesState();
}

class _RegisterPagesState extends State<RegisterPages> {
  //text editing controller
  final userNameController = TextEditingController();
  final emailController = TextEditingController();
  final passwordController = TextEditingController();
  final confirmPasswordController = TextEditingController();

  //sign user up
  void signUserUp() async {
    //show loading circle
    showDialog(
      context: context,
      builder: (context) {
        return const Center(
          child: CircularProgressIndicator(),
        );
      },
    );

    //try creating the user
    try {
      //check if password is confirmed
      if (passwordController.text == confirmPasswordController.text) {
        await FirebaseAuth.instance.createUserWithEmailAndPassword(
          email: emailController.text,
          password: passwordController.text,
        );
      } else {
        //show error message
        showError("Password don\'t match");
      }
      //pop the loading circle
      Navigator.pop(context);
    } on FirebaseAuthException catch (e) {
      Navigator.pop(context);
      //show error message
      showError(e.code);
    }
  }

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
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(
                  height: 10,
                ),
                const Icon(
                  Icons.lock,
                  size: 50,
                ),
                const SizedBox(
                  height: 10,
                ),
                Text(
                  'Let\'s create an account for you!.',
                  style: TextStyle(color: Colors.grey[700], fontSize: 16),
                ),
                const SizedBox(
                  height: 25,
                ),
                MyTextField(
                  controller: userNameController,
                  hintText: 'Username',
                  obscureText: false,
                ),
                const SizedBox(
                  height: 10,
                ),
                MyTextField(
                  controller: emailController,
                  hintText: 'Email',
                  obscureText: false,
                ),
                const SizedBox(
                  height: 10,
                ),

                MyTextField(
                  controller: passwordController,
                  hintText: 'Password',
                  obscureText: true,
                ),
                const SizedBox(
                  height: 10,
                ),
                //Confirm Password
                MyTextField(
                  controller: confirmPasswordController,
                  hintText: 'Confirm Password',
                  obscureText: true,
                ),

                const SizedBox(
                  height: 25,
                ),
                MyButton(
                  text: "Sign Up",
                  onTap: signUserUp,
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
                        onTap: () => AuthService().signInWithGoogle(context),
                        imagePath: 'lib/images/google.png'),
                    const SizedBox(
                      width: 10,
                    ),
                    // SquareTile(imagePath: 'lib/images/apple.png')
                  ],
                ),
                const SizedBox(
                  height: 50,
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      'Already have an account?',
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
                                builder: (context) => LoginPage()));
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
