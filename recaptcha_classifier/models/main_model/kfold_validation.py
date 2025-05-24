from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from recaptcha_classifier.features.evaluation.evaluate import evaluate_model
from recaptcha_classifier.models.main_model.HPoptimizer import HPOptimizer
from recaptcha_classifier.models.main_model.model_class import MainCNN


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

        :param data: Full dataset
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
        class_names = ['Car', 'Other', 'Cross', 'Bus', 'Hydrant',
                       'Palm', 'Tlight', 'Bicycle', 'Bridge', 'Stair',
                       'Chimney', 'Motorcycle']

        for fold_index, (train_indices, val_indices) in enumerate(kf.split(
                                                                 self.data)):
            print(f"\n--- Fold {fold_index + 1}/{self.k_folds} ---")

            train_data = [self.data[i] for i in train_indices]
            val_data = [self.data[i] for i in val_indices]

            train_loader = DataLoader(train_data, batch_size=32)
            val_loader = DataLoader(val_data, batch_size=32)

            self.hp_optimizer._trainer.train_loader = train_loader
            self.hp_optimizer._trainer.val_loader = val_loader

            opt_hp = self.hp_optimizer.optimize_hyperparameters()
            evaluated_models = []
            for _, row in opt_hp.iterrows():
                model = MainCNN(n_layers=int(row['layers']),
                                kernel_size=int(row['kernel_sizes']))
                metrics_result = evaluate_model(
                    model, val_loader, device=self.device, num_classes=12,
                    class_names=class_names, plot_cm=False
                )
                evaluated_models.append((model, metrics_result))

            self.best_models_per_fold.append(evaluated_models[:top_n_models])

    def get_all_best_models(self) -> list:
        """
        Get all best models from all folds.

        :return best_models_per_fold: The best models from all folds.
        """
        return self.best_models_per_fold

    def get_best_overall_model(self, metric_key='F1-score') -> tuple:
        """
        Finds the best model across all folds based on a specified metric.

        :param metric_key: The key in the metrics dictionary to use for
        selection.
        :return (best_model, best_metrics): Best model with its metrics
        results.
        """
        best_model = None
        best_metrics = None
        best_score = -float('inf')

        for fold_models in self.best_models_per_fold:
            for model, metrics_result in fold_models:
                score = metrics_result.get(metric_key, None)
                if score is not None and score > best_score:
                    best_model = model
                    best_metrics = metrics_result
                    best_score = score

        if best_model is None:
            print("Warning: No models found for selection.")

        return best_model, best_metrics
