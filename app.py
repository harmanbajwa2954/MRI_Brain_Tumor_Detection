import uuid
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.models import load_model
import os
import numpy as np
from flask import Flask, request, render_template, url_for
# Dynamically import the correct preprocess_input based on model type
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
import cv2
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
# Set this to match your model architecture
MODEL_ARCH = 'efficientnet'

if MODEL_ARCH == 'efficientnet':
    preprocess_input = efficientnet_preprocess_input
elif MODEL_ARCH == 'resnet50':
    preprocess_input = resnet50_preprocess_input
elif MODEL_ARCH == 'mobilenetv2':
    preprocess_input = mobilenetv2_preprocess_input
else:
    raise ValueError(
        f"Unknown MODEL_ARCH: {MODEL_ARCH}. Please set the correct preprocessing for your model.")

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
# Implementing XAI using gradcam


def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found")


def compute_eigencam(model, img_array):
    preds = model(img_array, training=False)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds))

    last_conv_name = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(last_conv_name).output]
    )

    conv_output = grad_model(img_array)[0].numpy()

    h, w, c = conv_output.shape
    reshaped = conv_output.reshape(h * w, c)
    reshaped -= reshaped.mean(axis=0)

    _, _, Vt = np.linalg.svd(reshaped, full_matrices=False)
    heatmap = reshaped @ Vt[0]
    heatmap = heatmap.reshape(h, w)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)

    return heatmap, class_idx, confidence


def generate_xai_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    orig = image.img_to_array(img).astype(np.uint8)

    arr = preprocess_input(orig.copy())
    arr = np.expand_dims(arr, axis=0)

    heatmap, _, _ = compute_eigencam(model, arr)

    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    pil_img = Image.fromarray(overlay)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()


# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = 'models/BT_EfB0_model.keras'
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
    # match your model's training size
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    # VGG16 preprocessing (-mean, BGR)
    arr = preprocess_input(arr)
    # suppress per-batch log spam
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = CLASS_LABELS[idx]
    conf = round(float(preds[idx]) * 100, 1)
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
    error = None

    if request.method == 'POST':
        f = request.files.get('file')
        if not f or f.filename == '':
            error = 'No file selected. Please choose an MRI image.'
        elif not allowed_file(f.filename):
            error = 'Invalid file type. Upload JPG, PNG, BMP, or TIFF.'
        elif model is None:
            error = 'Model file not found. Set the correct MODEL_PATH in app.py.'
        else:
            ext = f.filename.rsplit('.', 1)[1].lower()
            name = f'{uuid.uuid4().hex}.{ext}'
            path = os.path.join(app.config['UPLOAD_FOLDER'], name)
            f.save(path)
            try:
                label, conf, all_probs = predict(path)
                heatmap_img = generate_xai_image(path)
                result = {
                    'label':       label,
                    'confidence':  conf,
                    'description': CLASS_DESCRIPTIONS[label],
                    'all_probs':   all_probs,
                    'image_url':   url_for('static', filename=f'uploads/{name}'),
                    'heatmap':     heatmap_img,   # 👈 ADD THIS
                    'is_tumor':    label != 'No Tumor',
                }
            except Exception as e:
                error = f'Prediction failed: {e}'

    return render_template('diagnostics.html', result=result, error=error)


if __name__ == '__main__':
    app.run(debug=True)
