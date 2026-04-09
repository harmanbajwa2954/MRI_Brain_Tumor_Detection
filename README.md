# 🧠 NeuroScan AI — Brain Tumor Classification

>**Live Link 🔗** : [NeuroScan.AI](https://harmanbajwa-neurascan-ai.hf.space) 

> VGG16 Transfer Learning · Flask · MRI Classification · 4-Class Detection

---

## Project Structure

```
neuroscan/
├── app.py                        # Flask application & inference logic
├── Type_of_Brain_Tumor.ipynb                # VGG16 training script
├── requirements.txt
├── .env                          # Environment variables (create this)
├── model/
│   └── brain_tumor_vgg16.h5     # ← place your trained model here
├── static/
│   └── uploads/                  # auto-created, stores uploaded MRIs
└── templates/
    └── index.html                # Jinja2 template (main UI)
```

---

## Quick Start

### 1. Clone & create virtual environment

```bash
git clone <your-repo>
cd neuroscan
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place your model

Put your trained `.h5` file at:
```
model/brain_tumor_vgg16.h5
```

> **Don't have a model yet?** See the [Training](#training-the-model) section below.

### 4. Create a `.env` file

```env
SECRET_KEY=your-secret-key-here
FLASK_DEBUG=false
PORT=5000
```

### 5. Run the app

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Training the Model

### Dataset

Use the **[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** from Kaggle.

Expected directory structure after download:

```
dataset/
  Training/
    glioma/          # ~1321 images
    meningioma/      # ~1339 images
    notumor/         # ~1595 images
    pituitary/       # ~1457 images
  Testing/
    glioma/          # ~300 images
    meningioma/      # ~306 images
    notumor/         # ~405 images
    pituitary/       # ~300 images
```

### Run training

```bash
python train_model.py \
  --data_dir ./dataset \
  --epochs 20 \
  --ft_epochs 10 \
  --batch_size 32
```

**What happens:**
1. **Phase 1** — Backbone frozen, only the classification head is trained (20 epochs, LR=1e-3)
2. **Phase 2** — VGG16 block5 unfrozen for fine-tuning (10 epochs, LR=1e-5)
3. Best weights are saved automatically via `ModelCheckpoint`

Expected test accuracy: **~95–98%**

### Class label order (critical!)

Keras sorts folders alphabetically. The class index mapping will be:
```
glioma     → 0   (Glioma)
meningioma → 1   (Meningioma)
notumor    → 2   (No Tumor)
pituitary  → 3   (Pituitary)
```

`app.py` `CLASS_NAMES` is already set to `["Glioma", "Meningioma", "No Tumor", "Pituitary"]` to match this. **Do not change the order.**

---

## Deployment

### Option A — Gunicorn (production server)

```bash
gunicorn -w 2 -b 0.0.0.0:5000 \
  --timeout 120 \
  --access-logfile - \
  app:app
```

> Use `--workers 1` if deploying on a machine with limited RAM (TF model stays in a single process).

### Option B — Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```

```bash
docker build -t neuroscan .
docker run -p 5000:5000 -v $(pwd)/model:/app/model neuroscan
```

### Option C — Render / Railway / Fly.io

These platforms support Gunicorn deployments out of the box. Set the start command to:
```
gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 120 app:app
```

Set environment variables:
- `SECRET_KEY` — random secure string
- `FLASK_DEBUG` — `false`

---

## API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Renders the main UI |
| `/` | POST | Accepts `multipart/form-data` with `file` field, returns result page |
| `/health` | GET | JSON health check: `{status, model_loaded, model_file}` |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `neuroscan-dev-secret-change-in-prod` | Flask session secret |
| `FLASK_DEBUG` | `false` | Enable debug mode |
| `PORT` | `5000` | Server port |

---

## Medical Disclaimer

> ⚠️ **This tool is for research and educational purposes only.**
> It has not received FDA 510(k) clearance or CE marking.
> All clinical decisions remain the sole responsibility of licensed medical professionals.
> Never use this tool as a substitute for professional radiological assessment.
