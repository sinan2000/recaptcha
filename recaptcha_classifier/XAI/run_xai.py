import os

from recaptcha_classifier.XAI.explainability import Explainability
from recaptcha_classifier.server.load_model import load_main_model


if __name__ == '__main__':
    model = load_main_model()
    e = Explainability(model, n_samples=400)
    #e.gradcam_generate_explanations()
    e.evaluate_explanations_index(240)
    #e.overlay_image(index=241, img_opacity=0.75)
    # e.overlay_image(index=90, img_opacity=0.6)
    # e.overlay_image(index=218, img_opacity=0.7)
    # e.evaluate_explanations(n=400)
    # e.aggregate_eval()