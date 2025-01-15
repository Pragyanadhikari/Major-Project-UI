import tensorflow as tf
import os

# Correct path to the saved model
model_path = "/Users/pragyanadhikari/Desktop/majorProjectUI/backend/NULB.keras"  # Ensure the path and file name are correct

# Check if the model file exists
if os.path.exists(model_path):
    try:
        # Load the saved model
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        
        # Convert the model to TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False
        converter.experimental_enable_resource_variables = True
        tflite_model = converter.convert()

        
        # Save the TFLite model
        with open("NULB.tflite", "wb") as f:
            f.write(tflite_model)
            print("TensorFlow Lite model saved successfully as NULB.tflite")
    
    except Exception as e:
        print(f"Error during model conversion: {e}")
else:
    print(f"Model file {model_path} does not exist. Please check the path.")
