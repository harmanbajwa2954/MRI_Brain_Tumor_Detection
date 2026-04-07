import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Input, Flatten, Dropout, Dense

app = Flask(__name__)

# Configuration
# Create uploads folder if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



IMAGE_SIZE = 128
base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights=None) # weights=None because you will load your own
model = Sequential()
model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))  # Input layer
model.add(base_model)  # Add VGG16 base model
model.add(Flatten())  # Flatten the output of the base model
model.add(Dropout(0.3))  # Dropout layer for regularization
model.add(Dense(128, activation='relu'))  # Dense layer with ReLU activation
model.add(Dropout(0.2))  # Dropout layer for regularization
model.add(Dense(4, activation='softmax')) # Replace 4 with your number of classes

# 2. Load your saved weights
model.load_weights('tumor.weights.h5')

# Load your trained model (update path and image size as needed)
MODEL_PATH = 'brain_tumortype_model.keras'
model = load_model(MODEL_PATH)
IMAGE_SIZE = 128 

# Define your classes (Update these based on your specific dataset)
CLASSES = ['Pituitary', 'No Tumor', 'Meningioma', 'Glioma']

def predict_image(img_path):
    """Preprocesses the image and returns the model prediction."""
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # If your model expects standard scaling (like ResNet/VGG standard), add it here
    img_array = img_array / 255.0 
    
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    
    return CLASSES[class_index], confidence

@app.route('/')
def home():
    """Renders the Hero/Features page."""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Renders the upload page and handles predictions."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get prediction
            result, confidence = predict_image(filepath)
            
            return render_template('predict.html', result=result, confidence=confidence, user_image=filepath)
            
    return render_template('predict.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)