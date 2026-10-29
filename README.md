# Heart Disease Image Classification — Streamlit Deployment

This repository template is based on the supplied training code. The training script uses **TensorFlow/Keras EfficientNetB0** for **binary image classification**, input size **224 × 224**, two training stages (frozen backbone, then fine-tuning), evaluation plots, and Grad-CAM visualizations.

> **Research/education only:** This app is not a medical device and must not be used for diagnosis or treatment.

## Repository structure

```text
heart-disease-streamlit/
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore
└── models/
    ├── final_model.keras       # add after training
    └── class_names.json        # add after training
```

## 1. Prepare the trained model

The supplied script saves these files in Colab under `/content/outputs/`:

- `final_model.keras`
- `class_names.json`

After training, download those two files from Colab. Create a `models` folder in this repository and put both files there:

```text
models/final_model.keras
models/class_names.json
```

`class_names.json` is produced by the training code and preserves the folder-label order. Do not type class names from memory; use the generated file.

**Important:** The supplied file is Python source code despite its `.ipynb` filename. It starts with `from google.colab import drive` and uses Colab paths. It is not a standard notebook JSON file. For local use, save it as a `.py` script and adapt the data paths; for Colab, paste the code into notebook cells or upload it as a Python file.

## 2. Test locally

Use Python 3.11 for a compatible deployment environment. In a terminal from this project folder:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run the commands in Command Prompt instead:
```bat
.venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

The app expects a binary model with a single sigmoid output and 224 × 224 RGB input, matching the supplied training code.

## 3. Upload to GitHub

1. Create a new GitHub repository, for example `heart-disease-streamlit`.
2. Upload `app.py`, `requirements.txt`, `runtime.txt`, `README.md`, `.gitignore`, and the `models` folder.
3. Add `models/final_model.keras` and `models/class_names.json` only after you have downloaded them from training.
4. Do not upload the original multi-GB dataset or zip archives to GitHub. Keep datasets local/Drive-hosted and add them to `.gitignore`.
5. GitHub blocks ordinary files larger than 100 MB. If the `.keras` model is over that limit, use Git LFS or host the model on Hugging Face Hub / another stable file host and modify the app to download it. Check the host's size and bandwidth limits before deploying.
6. Never commit API tokens, passwords, or private data.

### Git commands (optional)

```bash
git init
git add app.py requirements.txt runtime.txt README.md .gitignore models/
git commit -m "Add Streamlit heart image classifier"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/heart-disease-streamlit.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and use the repository URL GitHub gives you.

## 4. Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**.
4. Select your repository, branch (`main`), and main file path (`app.py`).
5. Deploy and wait for dependencies to install.
6. Open the app URL and test with a permitted sample image.

If deployment fails:
- Check **Manage app → Logs** for the first error.
- Confirm the model and class-name files are present at the expected paths.
- Ensure the TensorFlow version is compatible with the version used to save the model. If needed, print `tf.__version__` in the training environment and pin a matching version in `requirements.txt`.
- TensorFlow can require substantial memory; large models or limited hosting resources may cause startup failure.
- Do not commit the full dataset just to make the app run. The deployed app only needs the trained model and class labels.

## 5. What the app does

- Accepts JPG, JPEG, PNG, BMP, and WEBP uploads.
- Resizes the image to 224 × 224 RGB.
- Loads the saved Keras model and class names.
- Displays the predicted class and sigmoid score.
- Shows a medical-use disclaimer.

## Important validation note

The training script detects a folder split if the data root has `train/` and `test/`; if no validation folder exists, it takes part of the training dataset for validation. If the data is not pre-split, it splits the root directory into a training partition and a validation/test partition. For a final scientific report, confirm that patient-level separation is valid and that the test set is not used for model selection.

## Important: Streamlit Cloud Python version

TensorFlow 2.19.1 supports Python 3.9–3.12, not Python 3.14. In Streamlit Community Cloud, open the app's **Manage app / Settings** and set the Python version to **3.12** (or 3.11) if that setting is available, then reboot/redeploy. The `runtime.txt` and `.python-version` files are included as additional version hints, but verify the Python version shown in the deployment logs. The logs must not say Python 3.14 when using TensorFlow 2.19.1.
