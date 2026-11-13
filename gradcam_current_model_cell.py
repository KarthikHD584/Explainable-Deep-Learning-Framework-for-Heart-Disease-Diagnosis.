# ============================================================
# GRAD-CAM FOR THE CURRENT MODEL  (replaces the old Grad-CAM cell)
# Paste this into a new cell AFTER the model has been trained / evaluated.
#
# Why the old cell stopped: it loaded all 9,512 test images into one float32
# array (9,512 x 224 x 224 x 3 x 4 bytes = 5.7 GB, then copied again), which
# exhausts Colab's RAM and silently restarts the kernel.
#
# This cell:
#   * goes through test_ds ONCE, batch by batch, and keeps only 2 random
#     examples per outcome (reservoir sampling), so memory stays tiny;
#   * computes Grad-CAM from the pre-sigmoid score (the logit), so the maps of
#     very confident predictions do not vanish (1 - sigmoid saturates);
#   * uses the 7x7x1280 top_activation output of EfficientNet-B0 (paper Eq. 2-3);
#   * saves fig5_gradcam.png in the 4 x 2 layout the revised paper expects
#     (rows: true positive, true negative, false negative, false positive).
#
# Note: test_ds is a take() of a reshuffled pool, so this pass sees a slightly
# different set of "test" images than the earlier evaluation cell (paper
# Section V-D). That is fine for showing examples, but these maps come from the
# image-level model. The patient-level maps for the paper come from the
# corrected notebook.
# ============================================================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

OUTPUT_DIR = "/content/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if "model" not in globals():
    path = os.path.join(OUTPUT_DIR, "final_model.keras")
    model = tf.keras.models.load_model(path, compile=False)
    print("Loaded", path)


def _parts(m):
    base = next(l for l in m.layers if l.name.startswith("efficientnet"))
    pred = m.get_layer("prediction")
    dense = [l for l in m.layers if isinstance(l, tf.keras.layers.Dense) and l.name != "prediction"][-1]
    return base, dense, pred


BASE, DENSE, PRED = _parts(model)


def gradcam(img):
    """img: (224, 224, 3) float32 in 0-255. Returns (cam 7x7 in [0, 1], probability of Sick).
    The head is applied with its own weights (dropout is the identity at inference)."""
    x = tf.keras.applications.efficientnet.preprocess_input(tf.convert_to_tensor(img[None], tf.float32))
    conv = tf.convert_to_tensor(BASE(x, training=False))            # (1, 7, 7, 1280)
    with tf.GradientTape() as tape:
        tape.watch(conv)
        h = tf.nn.relu(tf.matmul(tf.reduce_mean(conv, axis=(1, 2)), DENSE.kernel) + DENSE.bias)
        logit = tf.matmul(h, PRED.kernel) + PRED.bias
        prob = float(tf.sigmoid(logit)[0, 0])
        score = logit[:, 0] if prob >= 0.5 else -logit[:, 0]        # predicted class
    grads = tape.gradient(score, conv)
    w = tf.reduce_mean(grads, axis=(1, 2))                           # Eq. (2)
    cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1))[0]   # Eq. (3)
    return (cam / (tf.reduce_max(cam) + 1e-8)).numpy(), prob


def overlay(img, cam, alpha=0.4):
    cam_up = tf.image.resize(cam[..., None], img.shape[:2], method="bilinear").numpy()[..., 0]
    heat = plt.get_cmap("jet")(cam_up)[..., :3] * 255.0
    return np.clip(img * (1 - alpha) + heat * alpha, 0, 255).astype("uint8")


# ---- one pass over the test set, keep 2 random examples per outcome --------------
GROUPS = {"True positive": (1, 1), "True negative": (0, 0),
          "False negative": (1, 0), "False positive": (0, 1)}
K = 2
rng = np.random.default_rng(42)
keep = {g: [] for g in GROUPS}
seen = {g: 0 for g in GROUPS}
y_all, p_all = [], []

for images, labels in test_ds:
    probs = np.asarray(model.predict_on_batch(images)).ravel()
    labs = labels.numpy().ravel().astype(int)
    imgs = images.numpy()
    y_all.extend(labs); p_all.extend(probs)
    for img, y, p in zip(imgs, labs, probs):
        yhat = int(p >= 0.5)
        g = next(k for k, v in GROUPS.items() if v == (y, yhat))
        seen[g] += 1
        if len(keep[g]) < K:
            keep[g].append((img.copy(), y, p))
        else:                                   # reservoir sampling: every image equally likely
            j = rng.integers(seen[g])
            if j < K:
                keep[g][j] = (img.copy(), y, p)

y_all, p_all = np.array(y_all), np.array(p_all)
print(f"Images in this pass: {len(y_all):,} | accuracy {np.mean((p_all >= 0.5) == y_all):.4f}")
print({g: seen[g] for g in GROUPS})

# ---- figure: 4 rows (TP, TN, FN, FP) x 2 examples, sized for one IEEE column ------
CLASS_NAMES = CLASS_NAMES if "CLASS_NAMES" in globals() else ["Normal", "Sick"]
fig, axes = plt.subplots(4, 2, figsize=(3.5, 6.4), dpi=300)
rows = []
for r, g in enumerate(GROUPS):
    for c in range(K):
        ax = axes[r, c]; ax.axis("off")
        if c >= len(keep[g]):
            ax.text(0.5, 0.5, "no example", ha="center", va="center", fontsize=6, family="serif")
            continue
        img, y, p_saved = keep[g][c]
        cam, p = gradcam(img)
        assert abs(p - p_saved) < 1e-3, "Grad-CAM forward pass does not match the model's prediction"
        ax.imshow(overlay(img, cam))
        ax.set_title(f"{g}\nTrue {CLASS_NAMES[y]} · Pred {CLASS_NAMES[int(p >= 0.5)]} · p={p:.2f}",
                     fontsize=6, family="serif")
        rows.append(dict(group=g, true=CLASS_NAMES[y], pred=CLASS_NAMES[int(p >= 0.5)], prob_sick=p,
                         cam_peak_row=int(np.unravel_index(cam.argmax(), cam.shape)[0]),
                         cam_peak_col=int(np.unravel_index(cam.argmax(), cam.shape)[1])))
plt.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
fig.savefig(os.path.join(OUTPUT_DIR, "fig5_gradcam.png"), dpi=300)
plt.show()
pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "gradcam_examples.csv"), index=False)
print("Saved", os.path.join(OUTPUT_DIR, "fig5_gradcam.png"))
