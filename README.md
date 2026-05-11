---
title: NeuraScan.ai
emoji: 🦀
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🧠 NeuroScan AI — Brain Tumor Classification & 3D Segmentation

> EfficientNetB0 · U-Net · Flask · XAI (EigenCAM) · 3D BraTS Segmentation

---

## 🌟 Key Features

1. **Brain Tumor Classification (4-Class):** Detects Glioma, Meningioma, Pituitary tumors, or healthy brains ("No Tumor") from 2D MRI scans using a fine-tuned **EfficientNetB0** model.
2. **Explainable AI (XAI):** Automatically generates **EigenCAM** heatmaps overlaid on uploaded images to highlight the morphological features that influenced the model's predictions.
3. **3D MRI Tumor Segmentation:** Utilizes a custom **U-Net** architecture to process full 3D patient volumes from the BraTS dataset. Users can upload a `.zip` containing 4 MRI modalities (FLAIR, T1, T1ce, T2) to generate slice-by-slice tumor masks and visual overlays.

---

## 📂 Project Structure

```text
neuroscan/
├── app.py                        # Main Flask application & inference logic
├── requirements.txt              # Project dependencies
├── demo_patient.npy              # Pre-processed 3D volume for the segmentation demo
├── models/
│   ├── BT_EfB0_model.keras       # Trained EfficientNetB0 classification model
│   └── unet_brats_best.keras     # Trained U-Net segmentation model
├── static/
│   ├── script.js                 # Frontend interactivity
│   ├── style.css                 # Base styling
│   ├── segmentation_style.css    # Segmentation UI styling
│   └── uploads/                  # Auto-created, stores uploaded MRIs temporarily
└── templates/
    ├── base.html                 # Jinja2 base layout
    ├── index.html                # Main landing UI
    ├── diagnostics.html          # 2D Classification & XAI UI
    ├── segmentation.html         # 3D Segmentation & Slicer UI
    └── unet_architecture.html    # U-Net explanation page
    └── classification_architecture.html                   # Classification model explanation page
```
## Medical Disclaimer

> ⚠️ **This tool is for research and educational purposes only.**
> It has not received FDA 510(k) clearance or CE marking.
> All clinical decisions remain the sole responsibility of licensed medical professionals.
> Never use this tool as a substitute for professional radiological assessment.
