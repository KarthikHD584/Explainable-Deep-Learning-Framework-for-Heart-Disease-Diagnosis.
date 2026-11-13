# Heart Disease XAI

Explainable classification of cardiac MRI (CMR) slices as **Normal** or **Sick** (coronary artery disease)
with an ImageNet-pretrained EfficientNet-B0 and Grad-CAM heat maps. Training and evaluation run in Google
Colab; the trained models are served through a REST API, a web demo or a command-line tool.

> **Research prototype. Not a medical device and not for clinical use.** The models are trained on 30
> patients from one hospital and one scanner. Performance on other data is unknown.

## Project layout

```
notebooks/heart_xai_corrected_evaluation.ipynb   training + patient-level evaluation + Grad-CAM (Colab, GPU)
colab/gradcam_current_model_cell.py              Grad-CAM cell for the ORIGINAL notebook's model
paper/fill_paper.py                              puts the notebook's results into the revised paper
export_for_deployment.py                         packages trained models into models/ (run in Colab)
heart_xai/inference.py                           model loading, preprocessing, prediction, Grad-CAM
api.py                                           REST API (FastAPI)
app.py                                           web demo (Gradio; also works as a Hugging Face Space)
predict.py                                       command line: one image or a folder -> CSV + heat maps
tests/                                           automated tests (use a tiny stand-in model, no patient data)
Dockerfile, requirements*.txt
models/                                          deployment bundle goes here
```

## The workflow, start to finish

### 1. Train and evaluate (Colab, GPU)
Open `notebooks/heart_xai_corrected_evaluation.ipynb` in Colab, select **Runtime → Change runtime type →
GPU**, and run all cells. It uses the same model and schedule as the original notebook, but evaluates
with **patient-level 5-fold cross-validation** (each `Directory_N` folder is one patient), five seeds,
a threshold tuned on validation patients, and Grad-CAM on unseen patients. Each run takes about 3 h;
results are saved to `MyDrive/heart_xai_results` after every run, so a disconnected session can resume.

### 2. Finish the paper
Upload the revised paper `.docx` and `paper/fill_paper.py` to `MyDrive/heart_xai_results` and run the
notebook's last cell. Then write the three green notes (your reading of Fig. 5).

### 3. Export the deployment bundle (Colab)
```python
!python export_for_deployment.py --results /content/drive/MyDrive/heart_xai_results \
                                 --out /content/drive/MyDrive/heart_xai_deploy/models
```
This copies the five fold models (served as an **ensemble**: probabilities and heat maps are averaged),
writes `model_config.json` with the tuned threshold and the cross-validation results, writes
`requirements.lock.txt` with the exact TensorFlow/Keras versions, and verifies that each copied model
reloads with identical outputs. Download the folder into `models/` of this project.

To deploy a single model instead (for example `final_model.keras` from the original notebook):
```python
!python export_for_deployment.py --model /content/outputs/final_model.keras --out .../models
```

### 4. Run locally
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r models/requirements.lock.txt

uvicorn api:app --port 8000          # REST API, docs at http://localhost:8000/docs
python app.py                        # web demo at http://localhost:7860
python predict.py some_folder/ --gradcam --out results/
```

API example:
```bash
curl -F "file=@slice.jpg" "http://localhost:8000/predict?explain=true"
```
returns `label`, `prob_sick`, `threshold`, per-model probabilities, latency and the Grad-CAM overlay as
a base64 PNG. `GET /health` and `GET /model` report status and the model card information.

### 5. Deploy
**Docker (any cloud: Render, Railway, Google Cloud Run, AWS, Azure):**
```bash
docker build -t heart-xai .
docker run -p 8000:8000 heart-xai                    # API
docker run -p 7860:7860 heart-xai python app.py      # web demo
```
The image runs as a non-root user, has a health check, and pins TensorFlow to the training version when
`models/requirements.lock.txt` is present.

**Hugging Face Spaces (free web demo):** create a Space with SDK *Gradio*, upload `app.py`,
`heart_xai/`, `requirements.txt` and `models/` (use Git LFS for the `.keras` files). Keep the Space
**private**, or check the dataset licence first (see below).

### 6. Test
```bash
pip install -r requirements-dev.txt
pytest -q
```
The tests build a tiny stand-in model with the same layer names and head, so they check loading,
all input types, the ensemble, Grad-CAM (including that the map lands on a synthetic lesion) and every
API error path in a few seconds, without patient data.

## Model card (short)

| | |
|---|---|
| Task | Binary classification of a single 2-D CMR slice: Normal vs Sick (CAD) |
| Architecture | EfficientNet-B0 (ImageNet) + GAP → Dropout 0.3 → Dense 128 → Dropout 0.2 → Dense 1 (sigmoid) |
| Training data | CAD Cardiac MRI Dataset (Kaggle, D. Sharifrazi): 63,425 JPEG slices, 30 patients, Omid Hospital Tehran, Siemens 1.5 T |
| Evaluation | Patient-level 5-fold CV, mean ± SD over runs; see `models/model_config.json` |
| Explanation | Grad-CAM on the 7×7 top activation, for the predicted class |
| Input | One CMR slice (JPEG/PNG), any size; resized to 224×224 |
| Not suitable for | Clinical decisions; other modalities (echo, CT, X-ray); other scanners or sites without re-validation; patient-level diagnosis (the model scores single slices) |

## Data and licence

The Kaggle page lists the dataset licence as **"Unknown"**. Use it for non-commercial research only,
cite Khozeimeh et al., *Sci. Rep.* 12, 11178 (2022) and the Kaggle dataset, and do not redistribute the
images. Never commit patient images to this repository (`.gitignore` blocks JPEGs). Before making a demo
public, ask the dataset creator for permission. Choose a licence for your own code (for example MIT) and
add a `LICENSE` file.
