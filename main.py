from recaptcha_classifier.pipeline.simple_cnn_pipeline import (
    SimpleClassifierPipeline)
from recaptcha_classifier.pipeline.main_model_pipeline import (
    MainModelPipeline)


def main():
    pipeline1 = SimpleClassifierPipeline(epochs=1)
    pipeline1.run()

    pipeline2 = MainModelPipeline(epochs=1)
    pipeline2.run()


if __name__ == '__main__':
    main()
