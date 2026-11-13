"""Web demo (Gradio). Run:  python app.py   then open http://localhost:7860
Works unchanged as a Hugging Face Space (SDK: gradio, app_file: app.py).
"""
import os

import gradio as gr

from heart_xai import DISCLAIMER, HeartXAI

MODEL_DIR = os.getenv("MODEL_DIR", "models")
model = HeartXAI(MODEL_DIR)
info = model.info()


def run(image):
    if image is None:
        return None, None, "Upload a cardiac MRI slice first."
    pred = model.predict(image, explain=True)
    scores = {"Sick": pred.prob_sick, "Normal": 1 - pred.prob_sick}
    detail = (f"**Prediction:** {pred.label}  \n"
              f"**Probability of Sick:** {pred.prob_sick:.3f} (decision threshold {pred.threshold:.3f})  \n")
    if len(pred.model_probs) > 1:
        detail += "**Per-model probabilities:** " + ", ".join(f"{p:.3f}" for p in pred.model_probs) + "  \n"
    detail += ("\nThe heat map marks the image regions that pushed the model toward its predicted class "
               "(Grad-CAM, 7×7 resolution). It shows where the evidence is, not what it is.")
    return scores, pred.overlay, detail


evaluation = info.get("evaluation", {})
eval_md = ""
if evaluation:
    eval_md = "**Evaluation (patient-level cross-validation):** " + ", ".join(
        f"{k} {v}" for k, v in evaluation.items()) + "\n\n"

with gr.Blocks(title="Heart Disease XAI") as demo:
    gr.Markdown("# Explainable heart disease classification from cardiac MRI\n"
                "EfficientNet-B0 classifier (Normal vs Sick/CAD) for a single CMR slice, with a Grad-CAM heat "
                f"map. {len(model.models)} model(s) loaded.\n\n" + eval_md + f"⚠️ {DISCLAIMER}")
    with gr.Row():
        inp = gr.Image(type="pil", label="CMR slice (JPEG/PNG)")
        heat = gr.Image(type="numpy", label="Grad-CAM overlay")
    with gr.Row():
        scores = gr.Label(num_top_classes=2, label="Class probabilities")
        text = gr.Markdown()
    gr.Button("Classify", variant="primary").click(run, inputs=inp, outputs=[scores, heat, text])

if __name__ == "__main__":
    demo.launch(server_name=os.getenv("HOST", "0.0.0.0"), server_port=int(os.getenv("PORT", "7860")))
