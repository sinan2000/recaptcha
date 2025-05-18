from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from recaptcha_classifier.features.evaluation.evaluate import evaluate_model
from recaptcha_classifier.models.main_model.HPoptimizer import HPOptimizer


class KFoldValidation:
    """
    Class for performing k-Fold Cross-Validation
    integrated with hyperparameter optimization.
    """

    def __init__(self, data, k_folds: int,
                 hp_optimizer: HPOptimizer,
                 device=None) -> None:
        """
        Initialize the cross-validation setup.

        :param data: Full dataset (torch Dataset or list of samples)
        :param k_folds: Number of folds
        :param hp_optimizer: Instance of HPOptimizer
        :param device: Optional torch device
        """
        self.data = data
        self.k_folds = k_folds
        self.hp_optimizer = hp_optimizer
        self.device = device
        self.best_models_per_fold = []

    def run_cross_validation(self, top_n_models: int = 3) -> None:
        """
        Runs k-Fold Cross-Validation and hyperparameter optimization.

        :param top_n_models: Number of best models to keep from each fold.
        """
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        for fold_index, (train_indices, val_indices) in enumerate(kf.split(
                                                                 self.data)):
            print(f"\n--- Fold {fold_index + 1}/{self.k_folds} ---")

            train_data = [self.data[i] for i in train_indices]
            val_data = [self.data[i] for i in val_indices]

            self.hp_optimizer.trainer.train_loader = train_data

            best_models = self.hp_optimizer.optimize_hyperparameters()

            val_loader = DataLoader(val_data, batch_size=32)
            evaluated_models = []
            class_names = ['Car', 'Other', 'Cross', 'Bus', 'Hydrant',
                           'Palm', 'Tlight', 'Bicycle', 'Bridge', 'Stair',
                           'Chimney', 'Motorcycle']
            for model in best_models:
                # class labels for evaluate model function for cm
                metrics_result = evaluate_model(
                    model, val_loader, device=self.device, num_classes=12,
                    class_names=class_names, plot_cm=False
                )
                evaluated_models.append((model, metrics_result))

            # Keep only top_n_models
            self.best_models_per_fold.append(evaluated_models[:top_n_models])

    def get_all_best_models(self):
        """
        Get all best models from all folds.
        """
        return self.best_models_per_fold
