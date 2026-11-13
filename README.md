# Heart Disease XAI — Streamlit app

Upload one cardiac MRI slice → Normal/Sick prediction + Grad-CAM heat map.
Research prototype, not for clinical use.

## Repository layout (exactly this)
```
app.py
requirements.txt
.streamlit/config.toml
models/final_model.keras
```
`app.py` is self-contained: there is no `heart_xai` folder to import any more.

## Deploy on Streamlit Community Cloud
1. Put the files above in the root of your GitHub repository. Delete the old `app.py`, `heart_xai/`,
   `api.py` and `.gitattributes` if they are there. Commit `models/final_model.keras` normally (not Git LFS).
2. In Colab, run `import tensorflow as tf; print(tf.__version__)`. If it is not 2.21.0, change the
   `tensorflow-cpu==` line in `requirements.txt` to that version.
3. On share.streamlit.io: delete the current app, then **Create app** → choose the repo, branch `main`,
   main file `app.py` → **Advanced settings** → **Python 3.12** → Deploy.
   (The Python version can only be chosen when the app is created, which is why the old app has to be deleted.)
4. First start takes a few minutes (TensorFlow is large). The model is loaded once and cached.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
