import uuid
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.models import load_model
import os
import numpy as np
from flask import Flask, request, render_template, url_for, jsonify
# Dynamically import the correct preprocess_input based on model type
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
import keras.backend as K
import cv2
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
import tempfile
import zipfile
import glob
import nibabel as nib


# ---------------Segmentation Model-------------------------
@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def hybrid_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

segmentation_model = None
def get_segmentation_model():
    global segmentation_model
    if segmentation_model is None:
        print("Loading segmentation model for the first time...")
        segmentation_model = load_model('models/unet_brats_best.keras')
    return segmentation_model

# --------------Classification model---------------------
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
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # 16 MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'}

CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

CLASS_DESCRIPTIONS = {
    'Glioma':     'A malignant brain tumor arising from glial cells, typically in the cerebral hemispheres.',
    'Meningioma': 'Usually benign tumor originating from the meninges, often well-defined and slow-growing.',
    'Pituitary':  'Adenoma of the pituitary gland in the sella turcica, often causing hormonal imbalance.',
    'No Tumor':   'No detectable neoplastic lesion found. Healthy brain tissue observed in the scan.',
}


# ------------------Implementing XAI using EigneCam-------------------


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


# -------------------------Load model---------------------
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

# -------------------------------Helper functions for preprocessing for segmentation------------------------


def normalize_volume(volume):
    volume = volume.astype(np.float32)
    mask = volume > 0
    if not np.any(mask):
        return volume
    lower = np.percentile(volume[mask], 1)
    upper = np.percentile(volume[mask], 99)
    volume = np.clip(volume, lower, upper)
    mean = volume[mask].mean()
    std = volume[mask].std()
    volume = (volume - mean) / (std + 1e-8)
    volume[~mask] = 0
    return volume


def array_to_base64(img_array, is_mask=False):
    """Converts numpy arrays to base64 for the frontend Slicer."""
    if is_mask:
        # Red overlay with 40% opacity for tumor
        rgba = np.zeros((128, 128, 4), dtype=np.uint8)
        rgba[img_array[:, :, 0] > 0.5] = [255, 0, 0, 100]
        img = Image.fromarray(rgba, 'RGBA')
    else:
        # Grayscale MRI
        if img_array.max() == 0:  # Handle pure black slices
            img_array = np.zeros_like(img_array, dtype=np.uint8)
        else:
            img_array = ((img_array - img_array.min()) /
                         (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        img = Image.fromarray(img_array, 'L')

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def generate_overlay_image(base_img, mask_img):

    # Normalize MRI
    base_img = ((base_img - base_img.min()) /
                (base_img.max() - base_img.min() + 1e-8) * 255).astype(np.uint8)

    base_rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)

    # Binary mask
    binary_mask = (mask_img[:, :, 0] > 0.5).astype(np.uint8)

    # Red overlay
    overlay = np.zeros_like(base_rgb)
    overlay[:, :, 0] = binary_mask * 255

    blended = cv2.addWeighted(base_rgb, 0.75, overlay, 0.25, 0)

    pil_img = Image.fromarray(blended)

    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# ---------------------------------Routes----------------------------------------------------------


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


@app.route('/segment_demo', methods=['GET'])
def segment_demo():
    try:
        # 1. Load the pre-processed volume instantly
        volume_stack = np.load('demo_patient.npy')

        # 2. Run Inference
        segmodel = get_segmentation_model()
        predictions = segmodel.predict(volume_stack, batch_size=2)

        # 3. Package for Frontend
        response_data = []
        for i in range(volume_stack.shape[0]):
            base_mri = volume_stack[i, :, :, 0]  # Using FLAIR for visual base
            pred_mask = (predictions[i] > 0.5).astype(np.float32)
            # Only send slices with brain tissue to save bandwidth
            tumor_pixels = np.sum(pred_mask)

            if tumor_pixels > 5:
                response_data.append({
                    "slice_index": i,

                    "mri_image":
                    f"data:image/png;base64,{array_to_base64(base_mri, is_mask=False)}",

                    "mask_image":
                    f"data:image/png;base64,{array_to_base64(pred_mask, is_mask=True)}",

                    "overlay_image":
                    f"data:image/png;base64,{generate_overlay_image(base_mri, pred_mask)}",

                    "tumor_detected":
                    bool(tumor_pixels > 50),

                    "tumor_pixels":
                    int(tumor_pixels)
                })

        return jsonify({"status": "success", "slices": response_data})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/segment_upload', methods=['POST'])
def segment_upload():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    # Create a temporary directory that auto-deletes when the block ends
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, 'upload.zip')
        file.save(zip_path)

        try:
            # 1. Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # 2. Find Modalities dynamically
            all_nii_files = glob.glob(os.path.join(temp_dir, '**/*.nii*'), recursive=True)
            flair_file = [f for f in all_nii_files if 'flair' in f.lower()]
            t1ce_file = [f for f in all_nii_files if 't1ce' in f.lower()]
            t1_file = [f for f in all_nii_files if 't1' in f.lower() and 't1ce' not in f.lower()]
            t2_file = [f for f in all_nii_files if 't2' in f.lower()]

            if not all([flair_file, t1_file, t1ce_file, t2_file]):
                return jsonify({"status": "error", "message": "ZIP must contain flair, t1, t1ce, and t2 .nii files."}), 400
            
            print("Found 4 necessary MRI modalities.")

            # 3. Load & Normalize
            vol_flair = normalize_volume(nib.load(flair_file[0]).get_fdata())
            vol_t1 = normalize_volume(nib.load(t1_file[0]).get_fdata())
            vol_t1ce = normalize_volume(nib.load(t1ce_file[0]).get_fdata())
            vol_t2 = normalize_volume(nib.load(t2_file[0]).get_fdata())

            # 4. Build Volume Stack (155, 128, 128, 4)
            num_slices = vol_flair.shape[2]
            volume_stack = np.zeros((num_slices, 128, 128, 4), dtype=np.float32)

            for i in range(num_slices):
                volume_stack[i, :, :, 0] = cv2.resize(vol_flair[:, :, i], (128, 128), interpolation=cv2.INTER_AREA)
                volume_stack[i, :, :, 1] = cv2.resize(vol_t1[:, :, i], (128, 128), interpolation=cv2.INTER_AREA)
                volume_stack[i, :, :, 2] = cv2.resize(vol_t1ce[:, :, i], (128, 128), interpolation=cv2.INTER_AREA)
                volume_stack[i, :, :, 3] = cv2.resize(vol_t2[:, :, i], (128, 128), interpolation=cv2.INTER_AREA)

            # 5. Run Inference
            segmodel = get_segmentation_model()
            predictions = segmodel.predict(volume_stack, batch_size=2)

            # 6. Package Data
            response_data = []
            for i in range(num_slices):
                base_mri = volume_stack[i, :, :, 0]
                
                # DEFINED INSIDE THE LOOP: Generates the mask for the current slice 'i'
                pred_mask = (predictions[i] > 0.25).astype(np.float32)
                
                tumor_pixels = np.sum(pred_mask)

                if tumor_pixels > 5:
                    response_data.append({
                        "slice_index": i,
                        "mri_image": f"data:image/png;base64,{array_to_base64(base_mri, is_mask=False)}",
                        "mask_image": f"data:image/png;base64,{array_to_base64(pred_mask, is_mask=True)}",
                        "overlay_image": f"data:image/png;base64,{generate_overlay_image(base_mri, pred_mask)}",
                        "tumor_detected": bool(tumor_pixels > 5),
                        "tumor_pixels": int(tumor_pixels)
                    })

            return jsonify({"status": "success", "slices": response_data})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/segmentation', methods=['GET'])
def segmentation_page():
    # Renders the UI from the templates folder
    return render_template('segmentation.html')


if __name__ == '__main__':
    app.run(debug=True)
