import os
import io
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
import tensorflow as tf
from transformers import TFResNetModel, ResNetConfig
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import load_model
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load class names
with open('artifacts/class_names.json', 'r') as f:
    class_names = json.load(f)

# Define the custom layer for InceptionV3
class InceptionV3FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._base_model_initialized = False

    def build(self, input_shape):
        if not self._base_model_initialized:
            self.base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            self.base_model.trainable = False
            self._base_model_initialized = True
        super().build(input_shape)

    def call(self, inputs):
        x = tf.image.resize(inputs, (224, 224))  # Ensure input is resized to (299, 299)
        x = self.base_model(x, training=False)
        return x

# Function to load model with custom layer
def load_custom_inception_model(model_path):
    custom_objects = {'InceptionV3FeatureExtractor': InceptionV3FeatureExtractor}
    model = load_model(model_path, custom_objects=custom_objects)
    return model

# Example usage
try:
    model = load_custom_inception_model('artifacts/fashion_mnist_inception.h5')
except Exception as e:
    print(f"Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image for ResNet model"""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to numpy array
    image = np.array(image)
    
    # Resize and convert to RGB
    image = tf.expand_dims(image, -1)  # (28, 28, 1)
    image = tf.image.resize(image, [224, 224])  # (224, 224, 1)
    image = tf.image.grayscale_to_rgb(image)  # (224, 224, 3)
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()

@app.route('/')
def home():
    return render_template('index.html', class_names=class_names)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Read the file content first
        file_content = file.read()
        
        # Save original file
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        with open(original_path, 'wb') as f:
            f.write(file_content)
        
        # Process image from the bytes we already read
        image = Image.open(io.BytesIO(file_content))
        processed_image = preprocess_image(image)
        
        # Save processed image
        processed_image_pil = Image.fromarray((processed_image * 255).astype('uint8'))
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"processed_{file.filename}")
        processed_image_pil.save(processed_path)
        
        # Make prediction
        prediction = model.predict(np.expand_dims(processed_image, axis=0))
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        return jsonify({
            'original_image': original_path,
            'processed_image': processed_path,
            'class': class_names[predicted_class],
            'confidence': confidence,
            'all_predictions': {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)