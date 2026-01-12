# Adversarial Attack & Defense Framework (Web Prototype)

A professional, modern **demo UI** for the project:

**“Adversarial Attack and Defense Framework for Robust Image Classification.”**

This is a **front-end + backend prototype**. The backend exposes a pretrained **ResNet-18 ImageNet classifier** (1000 classes), and the front-end calls it to generate Top-5 predictions so you can demo the full workflow:

Upload an Image → Clean Classification → Generate Attack (FGSM/PGD) → Apply Defense → Compare Results

## Theme
- Dark + neon blue cyber-security aesthetic
- Futuristic cards, glow effects, clean presentation layout

## Features / Screens
- Home Dashboard
- Image Upload (drag & drop + preview)
- Clean Classification Output (Top-5)
- Attack Generation (FGSM / PGD + epsilon slider)
- Defense Mechanism (Adversarial Training, Denoising Filter, JPEG Compression, Gradient Masking)
- Final Comparison (clean vs adversarial vs defended + probability shift chart)
- About / Explanation page

## Run locally (Windows / PowerShell)

```powershell
npm install
npm run dev
```

Open the URL shown in the terminal (typically http://localhost:5173).

## Demo notes
- **Epsilon** controls the strength of the perturbation (0.01 – 0.30).
- Attacks/defenses are implemented using client-side canvas transforms.
- Predictions come from a real pretrained ImageNet model (ResNet-18) served by the Python backend.

## Built-in sample image
Use the **“Use Sample Image”** button on the Upload page.
- Sample asset name: `cifar10-sample-cat.svg` (any image will be classified into one of the 1000 ImageNet categories)

ULL SETUP (RUN ONLY ONCE)
Backend (ONE TIME)
cd backend
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt

Frontend (ONE TIME)
cd frontend
npm install

one click run: 
cd backend
.\.venv\Scripts\Activate
python -m uvicorn api:app --reload

