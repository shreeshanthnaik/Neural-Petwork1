import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# --- 1. Install this if you don't have it: pip install Pillow ---
from PIL import Image

print("Loading trained model...")
# Load your trained model
MODEL_PATH = 'cats_vs_dogs.h5'
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model {MODEL_PATH} loaded successfully.")

# Initialize the Flask app
app = Flask(__name__)

# --- 2. Define the image preprocessing function ---
def preprocess_image(image_path, target_size=(150, 150)):
    """
    Loads an image from a path, resizes it to 150x150,
    normalizes it, and adds a batch dimension.
    """
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert to numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Normalize pixel values
    img_array /= 255.0
    # Add a batch dimension (model expects 4D tensor: [batch_size, H, W, C])
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- 3. Define the main page route ---
@app.route('/', methods=['GET'])
def index():
    # Render the main HTML page
    return render_template('index.html')

# --- 4. Define the prediction route ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['image']
    
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        # We need a path to save it. Let's create an 'uploads' folder
        uploads_dir = os.path.join(app.root_path, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, filename)
        file.save(file_path)

        try:
            # Preprocess the image
            processed_image = preprocess_image(file_path)

            # Make a prediction
            prediction = model.predict(processed_image)
            
            # Get the raw prediction value
            score = prediction[0][0]

            # Interpret the result
            if score > 0.5:
                result_text = f"It's a Dog! (Confidence: {score*100:.2f}%)"
            else:
                result_text = f"It's a Cat! (Confidence: {(1-score)*100:.2f}%)"

        except Exception as e:
            print(f"Error during prediction: {e}")
            result_text = "Error: Could not process image."

        # Re-render the page with the result
        return render_template('index.html', prediction=result_text)

    return redirect(url_for('index'))

# --- 5. Run the app ---
if __name__ == '__main__':
    app.run(debug=True)