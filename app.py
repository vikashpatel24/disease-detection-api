import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify
from markupsafe import Markup
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
from flask_cors import CORS

# Load TFLite model
model_path = './disease_detect_model.tflite'  # Replace with your actual TFLite model path
tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
tflite_interpreter.allocate_tensors()

# Assuming disease_dic is imported from utils.diseases
from utils.diseases import disease_dic
# Calculate the size of disease_dic
# size_of_disease_dic = len(disease_dic)

# # Print the result
# print(f"Size of disease_dic: {size_of_disease_dic}")


# Disease names
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.route('/')
def home():
    return "Welcome to Disease Detection API!"

@app.route('/disease-detect', methods=['POST'])
def disease_prediction():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image provided'})
        
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'file is not in requests.files.get()'})
        
        file = file.read()  # Byte data
        image = read_file_as_image(file)
        image = tf.image.resize(image, [256, 256]).numpy()
        image = np.expand_dims(image, 0)  # 1D to 2D
        
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        tflite_interpreter.allocate_tensors()

        tflite_interpreter.set_tensor(input_details[0]['index'], image)
        tflite_interpreter.invoke()
        predictions = tflite_interpreter.get_tensor(output_details[0]['index'])

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = round(np.max(predictions[0]) * 100, 2)

        # Log image details
        image_name = request.files['file'].filename
        image_size = len(file)
        print(f"Image Name: {image_name}, Image Size: {image_size} bytes")

        data = {
            'confidence': float(confidence),
            'data': Markup(str(disease_dic[predicted_class]))
        }
        print(data)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
