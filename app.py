import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# --- 1. Load your trained YOLOv8 model ---
# Make sure this path is correct
MODEL_PATH = r"runs\detect\train\weights\best.pt"
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# Initialize the Flask app
app = Flask(__name__)

# --- 2. Configure upload and result folders ---
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static' # Flask serves static files from here
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

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
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # --- 5. Run the YOLOv8 prediction ---
            results = model.predict(source=file_path, save=True)
            
            # --- 6. Get the path to the result image ---
            result_save_dir = results[0].save_dir
            result_filename = filename # The saved image has the same name
            
            result_image_path = os.path.join(result_save_dir, result_filename)
            static_image_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            
            # --- THIS IS THE FIX ---
            # os.replace will overwrite the old file if it exists
            os.replace(result_image_path, static_image_path)
            
            # Pass the *filename* to the HTML template
            return render_template('index.html', result_image=result_filename)

        except Exception as e:
            print(f"An error occurred: {e}")
            # If an error happens, tell the user on the webpage
            return render_template('index.html', prediction=f"Error: {e}")

    return redirect(url_for('index'))

# --- 6. Run the app ---
if __name__ == '__main__':
    app.run(debug=True)