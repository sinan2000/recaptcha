from src.XAI.explainability import Explainability
from src.server.load_model import load_main_model


if __name__ == '__main__':
    # Trained:
    model = load_main_model()
    # Random:
    # model = MainCNN(n_layers=1, kernel_size=1)

    # XAI
    e = Explainability(model, n_samples=400)
    # e.run(eval_percent_samples=...)
    # e.evaluate_explanations_index(241)    # low confidence example
