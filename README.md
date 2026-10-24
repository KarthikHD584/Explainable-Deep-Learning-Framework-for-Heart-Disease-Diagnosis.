# 🫀 Explainable Deep Learning Framework for Heart Disease Diagnosis

Image-based heart disease classification with **EfficientNetB0 transfer learning** and
**Grad-CAM** explanations, served as a **Streamlit** web app.

> ⚠️ **Disclaimer:** research/educational prototype. Not a medical device. Not validated for
> clinical use. Do not use it to make health decisions.

**Live demo:** `https://<your-app-name>.streamlit.app` *(add after deploying)*

---

## What it does

1. Trains a binary image classifier (frozen backbone, then fine-tuning) on the heart disease image dataset.
2. Evaluates on a held-out, stratified test set (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).
3. Explains each prediction with a Grad-CAM heatmap showing which image regions drove the decision.
4. Serves the model in a web app: upload an image, get a prediction, confidence and heatmap.

## Results

Fill these in from `model/metrics.json` after training (do not guess):

| Metric | Test set |
|---|---|
| Accuracy | _ |
| Precision | _ |
| Recall | _ |
| F1 | _ |
| ROC-AUC | _ |

Figures (confusion matrix, ROC, Grad-CAM grid) are in `assets/`.

## Repository structure

```
.
├── app.py                      # Streamlit app (prediction + Grad-CAM)
├── requirements.txt
├── model/
│   ├── final_model.keras       # trained model (from Colab)
│   ├── class_names.json
│   └── metrics.json
├── assets/                     # confusion_matrix.png, roc_curve.png, gradcam_grid.png
├── notebooks/
│   └── heart_disease_dl_colab.ipynb   # full training pipeline
└── .streamlit/config.toml
```

The raw dataset (~3 GB) is **not** stored in the repo.

## 1. Train the model (Google Colab)

1. Upload `notebooks/heart_disease_dl_colab.ipynb` to Colab and set the runtime to **GPU**.
2. Put the dataset zip in Drive and set `ZIP_PATH` in the notebook.
3. Run all cells. **Read the folder tree printed in section 4** and confirm the classes and splits look right.
4. The last cell copies these files to `MyDrive/heart_disease_xai_export/`:
   `final_model.keras`, `class_names.json`, `metrics.json`, and the three PNG figures.
   It also prints the TensorFlow version to pin.

## 2. Add the files to this repo

- `final_model.keras`, `class_names.json`, `metrics.json` → `model/`
- `confusion_matrix.png`, `roc_curve.png`, `gradcam_grid.png` → `assets/`
- In `requirements.txt`, change `tensorflow-cpu>=2.16` to the exact version printed by the notebook (e.g. `tensorflow-cpu==2.xx.x`).

File size: GitHub warns above 50 MB and rejects files above 100 MB. If `final_model.keras` is larger, enable
Git LFS (`git lfs install`, then uncomment the line in `.gitattributes`).

## 3. Run locally (optional)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 4. Push to GitHub

```bash
git init
git add .
git commit -m "Heart disease XAI: model + Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

(Create the empty repo on github.com first.) Keep the repo public to use the free Community Cloud tier freely.

## 5. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub (authorize access to your repo).
2. Click **Create app** (or **New app**) and pick this repository, branch `main`, main file path `app.py`.
3. Click **Advanced settings** and choose **Python 3.12** (3.11 also works with TensorFlow).
   `runtime.txt` is not supported on Community Cloud; the version is set only in this dialog or the app settings.
4. Click **Deploy**. The first build installs TensorFlow and takes a few minutes.
5. Every `git push` to `main` redeploys the app automatically.

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| App shows "This app has gone over its resource limits" | TensorFlow + the model is memory-hungry. Reboot the app, keep `tensorflow-cpu`, and upload one image at a time. Community Cloud gives roughly 0.7 GB guaranteed, up to ~2.7 GB when available. If it keeps crashing, host elsewhere (e.g. Hugging Face Spaces) or convert the model to a lighter runtime. |
| `ValueError` / error loading `final_model.keras` | TensorFlow/Keras version differs from the one used in Colab. Pin the exact version from the notebook's last cell. |
| Build fails installing TensorFlow | Wrong Python version. Select 3.12 (or 3.11) in Advanced settings. |
| "Model files not found" | `model/final_model.keras` or `model/class_names.json` was not committed. |
| `git push` rejected for file size | Model above 100 MB; use Git LFS. |

## How Grad-CAM is computed

The backbone is a nested model, so the app builds a feature model (backbone input → last conv maps and
backbone output), applies the classification head manually inside a `GradientTape`, and weights the conv
maps by the pooled gradients of the **predicted class** score. Warm regions influenced the prediction most.
A heatmap shows what the model relied on; it is not proof that the model is clinically right.

## Limitations

- Binary classifier trained on a single dataset; performance on other scanners, hospitals or populations is unknown.
- If the dataset has several images per patient, a random image-level split can inflate test scores; split by patient.
- No calibration, uncertainty estimation or external validation yet.
- Predictions are not clinical advice.

## License

Add a license (e.g. MIT) and cite the dataset source/licence here.
