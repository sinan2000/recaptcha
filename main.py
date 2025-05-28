from recaptcha_classifier.pipeline.main_model_pipeline import MainClassifierPipeline
from recaptcha_classifier.pipeline.simple_cnn_pipeline import SimpleClassifierPipeline


def main():
    pipeline1 = SimpleClassifierPipeline()
    pipeline1.run(save_train_checkpoints=False)

    # pipeline2 = MainClassifierPipeline(epochs=1, k_folds=2)
    # pipeline2.run(save_train_checkpoints=False)


if __name__ == '__main__':
    main()