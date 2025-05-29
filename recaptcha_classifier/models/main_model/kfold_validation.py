import pandas as pd
from sklearn.model_selection import KFold
from sympy.printing.pytorch import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from recaptcha_classifier.features.evaluation.evaluate import evaluate_model
from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.models.main_model.model_class import MainCNN
from recaptcha_classifier.constants import MODELS_FOLDER


class KFoldValidation:
    """
    Class for performing k-Fold Cross-Validation.
    """

    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 k_folds: int,
                 device=None) -> None:
        """
        Initialize the cross-validation setup.

        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        :param k_folds: Number of folds
        :param hp_optimizer: Instance of HPOptimizer
        :param device: Optional torch device
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.k_folds = k_folds
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")

    def run_cross_validation(self,
                             hp: list,
                             save_checkpoints: bool = True,
                             load_checkpoints: bool = False,
                             batch_size: int = 32) -> None:
        """
        Runs k-Fold Cross-Validation and hyperparameter optimization.

        :param hp: list of hyperparameters to optimize
        :param load_checkpoints: boolean to load checkpoints
        :param batch_size: size of batches in fold data loaders
        :param save_checkpoints: boolean flag to save checkpoints
        """

        # Concatenating all data indices from both loaders
        train_indices = list(range(len(self.train_loader.dataset)))
        val_indices = list(range(len(self.val_loader.dataset)))
        all_indices = train_indices + val_indices
        dataset = self.train_loader.dataset

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        results = []

        n_layers, kernel_sizes, learning_rates = hp

        for fold_index, (t_idx, val_idx) in enumerate(kf.split(all_indices)):

            print(f"\n--- Fold {fold_index + 1}/{self.k_folds} ---")

            train_subset = Subset(dataset, [all_indices[i]
                                            for i in t_idx])
            val_subset = Subset(dataset, [all_indices[i] for i in val_idx])

            fold_train_loader = DataLoader(train_subset,
                                           batch_size=batch_size,
                                           shuffle=False)
            fold_val_loader = DataLoader(val_subset,
                                         batch_size=batch_size,
                                         shuffle=False)

            model = MainCNN(n_layers=n_layers, kernel_size=kernel_sizes)

            trainer = Trainer(fold_train_loader, fold_val_loader,
                              epochs=20, save_folder=MODELS_FOLDER,
                              device=self.device)

            trainer.train(model,
                          lr=learning_rates,
                          save_checkpoint=save_checkpoints,
                          load_checkpoint=load_checkpoints)

            metrics = evaluate_model(model, fold_val_loader,
                                     device=self.device)
            metrics.pop('Confusion Matrix')
            metrics["fold"] = fold_index + 1
            results.append(metrics)

        df_results = pd.DataFrame(results)

        df_results.to_csv(f"{MODELS_FOLDER}/kfold_results.csv", index=False)
        self.print_summary(df_results)
        self.plot_results(df_results)

    @staticmethod
    def print_summary(results: pd.DataFrame) -> None:
        """
        Prints a summary of the cross-validation results.
        """
        print("\n~~ Cross-Validation Summary ~~")
        print(results.round(3))

        res = results.drop(columns=["fold"])

        means = res.mean()
        print("\nMean Results Across Folds:")
        print(means.round(3))

        stds = res.std()
        print("\nStandard Deviation Across Folds:")
        print(stds.round(3))

    @staticmethod
    def plot_results(results: pd.DataFrame) -> None:
        """
        Plots the results of the cross-validation.
        """
        metrics = [col for col in results.columns if col != 'fold']
        mean_vals = results[metrics].mean()
        std_vals = results[metrics].std()

        fig, ax = plt.subplots(figsize=(10, 6))
        mean_vals.plot(kind="bar", yerr=std_vals, capsize=5, ax=ax)

        ax.set_title("Cross-Validation Metrics (mean +- std)")
        ax.set_ylabel("Score")
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y')
        plt.show()
