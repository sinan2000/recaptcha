import os

from recaptcha_classifier.constants import BASE_DIR
from recaptcha_classifier.server.load_model import load_main_model

model = load_main_model()

if __name__ == '__main__':
    model = load_main_model()