"""Package trained models into the deployment bundle (models/ folder) — run this in Colab.

Option A, recommended: the five patient-level runs of heart_xai_corrected_evaluation.ipynb (ensemble)
    !python export_for_deployment.py --results /content/drive/MyDrive/heart_xai_results \
                                     --out /content/drive/MyDrive/heart_xai_deploy/models

Option B: one model, e.g. /content/outputs/final_model.keras from the original notebook
    !python export_for_deployment.py --model /content/outputs/final_model.keras --threshold 0.5 \
                                     --out /content/drive/MyDrive/heart_xai_deploy/models

Writes model_config.json, the .keras files and requirements.lock.txt (the exact TensorFlow/Keras versions
used for training; install these in production so the models load identically). Then reloads every model
and checks that it gives the same output as the original.
"""
import argparse
import datetime as dt
import glob
import json
import os
import shutil
import sys

import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--results", help="RESULTS_DIR of the corrected notebook (run*/final_model.keras)")
g.add_argument("--model", help="a single .keras model")
ap.add_argument("--threshold", type=float, default=None, help="decision threshold (single-model mode)")
ap.add_argument("--out", required=True)
ap.add_argument("--version", default=dt.date.today().strftime("%Y.%m.%d"))
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
cfg = {"model_version": args.version, "class_names": ["Normal", "Sick"], "img_size": [224, 224],
       "tensorflow": tf.__version__, "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "",
       "trained_on": "CAD Cardiac MRI Dataset (Kaggle, D. Sharifrazi), 30 patients, JPEG CMR slices",
       "exported": dt.datetime.now().isoformat(timespec="seconds")}
try:
    import keras
    cfg["keras"] = keras.__version__
except ImportError:
    pass

sources = []
if args.results:
    runs = sorted(glob.glob(os.path.join(args.results, "run*_seed*", "final_model.keras")))
    if not runs:
        sys.exit(f"No run*/final_model.keras in {args.results}")
    for k, src in enumerate(runs):
        sources.append((src, f"model_run{k}.keras"))
    import pandas as pd
    per_run = pd.read_csv(os.path.join(args.results, "per_run_metrics.csv"))
    cfg["threshold"] = round(float(per_run["thr"].mean()), 4)
    cfg["threshold_rule"] = ("mean of the per-run thresholds chosen on validation patients for the target "
                             "sensitivity (see paper Section III-E)")
    vals = json.load(open(os.path.join(args.results, "paper_values.json"), encoding="utf-8"))
    cfg["evaluation"] = {"accuracy_%": vals["NEW05_ACC"], "roc_auc": vals["NEW05_AUC"],
                         "sensitivity_%_at_threshold": vals["NEWT_SENS"],
                         "specificity_%_at_threshold": vals["NEWT_SPEC"],
                         "protocol": "patient-level 5-fold CV, mean ± SD"}
else:
    sources.append((args.model, "model.keras"))
    cfg["threshold"] = 0.5 if args.threshold is None else args.threshold
    cfg["threshold_rule"] = "fixed" if args.threshold is None else "given at export"
    cfg["evaluation_note"] = ("Single model. If it comes from the original image-level split, its test scores "
                              "are optimistic (patients overlap between training and test).")

cfg["models"] = [dst for _, dst in sources]
probe = np.random.default_rng(0).uniform(0, 255, (2, 224, 224, 3)).astype("float32")
for src, dst in sources:
    shutil.copy(src, os.path.join(args.out, dst))
    a = tf.keras.models.load_model(src, compile=False)(probe, training=False).numpy()
    b = tf.keras.models.load_model(os.path.join(args.out, dst), compile=False)(probe, training=False).numpy()
    assert np.allclose(a, b, atol=1e-6), f"{dst}: reloaded model differs"
    print(f"copied + verified {dst}")

with open(os.path.join(args.out, "model_config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)

import importlib.metadata as md
pins = []
for pkg in ("tensorflow", "tensorflow-cpu", "keras", "numpy", "pillow", "matplotlib"):
    try:
        v = md.version(pkg)
    except md.PackageNotFoundError:
        continue
    pins.append(f"tensorflow=={v}" if pkg == "tensorflow-cpu" else f"{pkg}=={v}")
pins = list(dict.fromkeys(pins))
with open(os.path.join(args.out, "requirements.lock.txt"), "w") as f:
    f.write("# Versions used for training/export. Install these for serving.\n" + "\n".join(pins) + "\n")
print(json.dumps(cfg, indent=2, ensure_ascii=False))
print("\nBundle written to", args.out)
