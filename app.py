from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
import uuid

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Ensure the static directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = 'models/colon_cnn.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Class mapping (same as used during training)
class_names = ['0_normal', '1_ulcerative_colitis', '2_polyps', '3_esophagitis']

def prepare_image(image_path):
    img = Image.open(image_path).resize((224, 224)).convert('RGB')
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded.', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected.', 400

    # Sanitize and uniquely name file
    original_filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    # Save file to static directory
    file.save(save_path)

    # Prepare and predict
    img = prepare_image(save_path)
    prediction = model.predict(img)
    raw_label = class_names[np.argmax(prediction)]

    # Clean up label: remove prefix, convert to title case
    predicted_class = raw_label.split('_', 1)[-1].replace('_', ' ').title()

    # Get image URL for rendering
    image_url = url_for('static', filename=unique_filename)

    return render_template('index.html', prediction=predicted_class, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
