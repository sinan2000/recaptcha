import os

from recaptcha_classifier.XAI.explainability import Explainability
from recaptcha_classifier.server.load_model import load_main_model


if __name__ == '__main__':
    model = load_main_model()
    e = Explainability(model, n_samples=400)
    #e.gradcam_generate_explanations()
    #e.overlay_image(index=0, img_opacity=0.8)
    e.overlay_image(index=2, img_opacity=0.7)
    e.overlay_image(index=2, img_opacity=0.75)
    e.overlay_image(index=2, img_opacity=0.8)
    #e.evaluate_explanations()