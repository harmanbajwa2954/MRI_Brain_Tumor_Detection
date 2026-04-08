import os
import numpy as np
from flask import Flask, request, render_template, url_for
# Dynamically import the correct preprocess_input based on model type
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input

# Set this to match your model architecture
MODEL_ARCH = 'vgg16'  # options: 'vgg16', 'resnet50', 'mobilenetv2'

if MODEL_ARCH == 'vgg16':
    preprocess_input = vgg16_preprocess_input
elif MODEL_ARCH == 'resnet50':
    preprocess_input = resnet50_preprocess_input
elif MODEL_ARCH == 'mobilenetv2':
    preprocess_input = mobilenetv2_preprocess_input
else:
    raise ValueError(f"Unknown MODEL_ARCH: {MODEL_ARCH}. Please set the correct preprocessing for your model.")
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'}

CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

CLASS_DESCRIPTIONS = {
    'Glioma':     'A malignant brain tumor arising from glial cells, typically in the cerebral hemispheres.',
    'Meningioma': 'Usually benign tumor originating from the meninges, often well-defined and slow-growing.',
    'Pituitary':  'Adenoma of the pituitary gland in the sella turcica, often causing hormonal imbalance.',
    'No Tumor':   'No detectable neoplastic lesion found. Healthy brain tissue observed in the scan.',
}

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = 'models/brain_tumortype_model.keras'
try:
    model = load_model(MODEL_PATH)
    print(f'[OK] Model loaded from {MODEL_PATH}')
except Exception as e:
    model = None
    import traceback
    print(f'[ERROR] Could not load model from "{MODEL_PATH}":')
    traceback.print_exc()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(img_path):
    img  = image.load_img(img_path, target_size=(128, 128))   # match your model's training size
    arr  = image.img_to_array(img)
    arr  = np.expand_dims(arr, axis=0)
    arr  = preprocess_input(arr)                              # VGG16 preprocessing (-mean, BGR)
    preds = model.predict(arr, verbose=0)[0]                  # suppress per-batch log spam
    idx  = int(np.argmax(preds))
    label = CLASS_LABELS[idx]
    conf  = round(float(preds[idx]) * 100, 1)
    all_probs = sorted(
        [{'name': CLASS_LABELS[i], 'prob': round(float(preds[i]) * 100, 1),
          'predicted': i == idx} for i in range(len(CLASS_LABELS))],
        key=lambda x: x['prob'], reverse=True
    )
    return label, conf, all_probs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnostics', methods=['GET', 'POST'])
def index():
    result = None
    error  = None

    if request.method == 'POST':
        f = request.files.get('file')
        if not f or f.filename == '':
            error = 'No file selected. Please choose an MRI image.'
        elif not allowed_file(f.filename):
            error = 'Invalid file type. Upload JPG, PNG, BMP, or TIFF.'
        elif model is None:
            error = 'Model file not found. Set the correct MODEL_PATH in app.py.'
        else:
            ext  = f.filename.rsplit('.', 1)[1].lower()
            name = f'{uuid.uuid4().hex}.{ext}'
            path = os.path.join(app.config['UPLOAD_FOLDER'], name)
            f.save(path)
            try:
                label, conf, all_probs = predict(path)
                result = {
                    'label':       label,
                    'confidence':  conf,
                    'description': CLASS_DESCRIPTIONS[label],
                    'all_probs':   all_probs,
                    'image_url':   url_for('static', filename=f'uploads/{name}'),
                    'is_tumor':    label != 'No Tumor',
                }
            except Exception as e:
                error = f'Prediction failed: {e}'

    return render_template('diagnostics.html', result=result, error=error)


if __name__ == '__main__':
    app.run(debug=True)