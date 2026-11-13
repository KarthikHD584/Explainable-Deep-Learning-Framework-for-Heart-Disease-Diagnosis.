"""Command-line prediction for one image or a whole folder.

  python predict.py path/to/slice.jpg
  python predict.py path/to/folder --out results/ --gradcam

Writes predictions.csv (and Grad-CAM overlays with --gradcam) to --out.
"""
import argparse
import csv
import os
import sys

from PIL import Image

from heart_xai import DISCLAIMER, HeartXAI

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="image file or folder (searched recursively)")
    ap.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "models"))
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--gradcam", action="store_true", help="save Grad-CAM overlays")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        files = sorted(os.path.join(r, f) for r, _, fs in os.walk(args.input) for f in fs if f.lower().endswith(EXTS))
    else:
        files = [args.input]
    if not files:
        sys.exit("No images found.")

    model = HeartXAI(args.model_dir)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for i, path in enumerate(files, 1):
        try:
            pred = model.predict(path, explain=args.gradcam)
        except ValueError as e:
            print(f"skip {path}: {e}", file=sys.stderr)
            continue
        row = {"file": path, "label": pred.label, "prob_sick": f"{pred.prob_sick:.6f}",
               "threshold": pred.threshold}
        if args.gradcam:
            name = os.path.splitext(os.path.relpath(path, args.input if os.path.isdir(args.input)
                                                    else os.path.dirname(path)))[0].replace(os.sep, "__")
            cam_path = os.path.join(args.out, f"{name}_gradcam.png")
            Image.fromarray(pred.overlay).save(cam_path)
            row["gradcam"] = cam_path
        rows.append(row)
        print(f"[{i}/{len(files)}] {pred.label:6s} p={pred.prob_sick:.3f}  {path}")

    out_csv = os.path.join(args.out, "predictions.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out_csv}\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
