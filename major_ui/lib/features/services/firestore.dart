import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class FirestoreService {
  final FirebaseFirestore _db = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;

  // Method to add a stock
  Future<void> addStock(
      String company, int numberOfStocks, String userId) async {
    try {
      await _db.collection('stocks').add({
        'company': company,
        'number_of_stocks': numberOfStocks,
        'user_id': userId,
        'timestamp': FieldValue.serverTimestamp(),
      });
    } catch (e) {
      throw Exception('Failed to add stock: $e');
    }
  }

  // Method to update an existing stock
  Future<void> updateStock(
      String docId, String company, int numberOfStocks) async {
    try {
      await _db.collection('stocks').doc(docId).update({
        'company': company,
        'number_of_stocks': numberOfStocks,
        'timestamp': FieldValue.serverTimestamp(),
      });
    } catch (e) {
      throw Exception('Failed to update stock: $e');
    }
  }

  // Method to delete a stock
  Future<void> deleteStock(String docId) async {
    try {
      await _db.collection('stocks').doc(docId).delete();
    } catch (e) {
      throw Exception('Failed to delete stock: $e');
    }
  }

  // Method to get the stream of stocks for a specific user
  Stream<QuerySnapshot> getStockStream() {
    String? userId = _auth.currentUser?.uid;
    if (userId == null) {
      throw Exception('User is not logged in');
    }

    return _db
        .collection('stocks')
        .where('user_id', isEqualTo: userId) // Filter by user_id
        .orderBy('timestamp')
        .snapshots();
  }

  // Method to get a single stock document by docId
  Future<DocumentSnapshot> getStockById(String docId) async {
    try {
      return await _db.collection('stocks').doc(docId).get();
    } catch (e) {
      throw Exception('Failed to get stock by ID: $e');
    }
  }
}
