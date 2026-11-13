"""Heart Disease XAI: EfficientNet-B0 classifier for cardiac MRI slices with Grad-CAM explanations."""
from .inference import HeartXAI, Prediction, overlay, to_png_bytes

__version__ = "1.0.0"
__all__ = ["HeartXAI", "Prediction", "overlay", "to_png_bytes", "__version__"]

DISCLAIMER = ("Research prototype. Not a medical device and not for clinical use. Trained on 30 patients from "
              "one hospital and one scanner (CAD Cardiac MRI Dataset); results on other data are unknown.")
