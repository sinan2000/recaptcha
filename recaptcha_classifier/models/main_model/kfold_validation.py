import pandas as pd
from sklearn.model_selection import KFold
from sympy.printing.pytorch import torch
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
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._best_models = pd.DataFrame()

    def run_cross_validation(self,
                             save_checkpoints: bool = True) -> None:
        """
        Runs k-Fold Cross-Validation and hyperparameter optimization.

        :param save_checkpoints: boolean flag to save checkpoints
        :param top_n_models: Number of best models to keep from each fold.
        """
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        class_names = ['BICYCLE', 'BRIDGE', 'BUS', 'CAR', 'CHIMNEY',
                       'CROSSWALK', 'HYDRANT', 'MOTORCYCLE', 'PALM',
                       'STAIR', 'TRAFFIC_LIGHT', 'OTHER']

        for fold_index, (train_indices, val_indices) \
                in enumerate(kf.split(self.data)):
            print(f"\n--- Fold {fold_index + 1}/{self.k_folds} ---")

            train_data = [self.data[i] for i in train_indices]
            val_data = [self.data[i] for i in val_indices]

            train_loader = DataLoader(train_data, batch_size=32)
            val_loader = DataLoader(val_data, batch_size=32)

            self.hp_optimizer._trainer.train_loader = train_loader
            self.hp_optimizer._trainer.val_loader = val_loader

            optimized_hp_dataframe = self.hp_optimizer.optimize_hyperparameters(save_checkpoints=save_checkpoints)
            evaluated_models = {'Accuracy': [],
                                'F1-score': [],
                            }
            evaluated_models = pd.DataFrame(evaluated_models)
            for _, row in optimized_hp_dataframe.iterrows():
                model = MainCNN(n_layers=int(row['layers']),
                                kernel_size=int(row['kernel_sizes']),
                                num_classes=12)
                metrics_result = evaluate_model(
                    model, val_loader, device=self.device, num_classes=12,
                    class_names=class_names, plot_cm=False
                )
                # removing confusion matrix to keep dimensions balanced
                metrics_result.pop('Confusion Matrix')
                metrics_result = pd.DataFrame(metrics_result, index=[0])
                evaluated_models = pd.concat([evaluated_models, metrics_result])

            optimized_hp_dataframe = pd.concat([optimized_hp_dataframe, evaluated_models])

            if not self._best_models.empty:
                self._best_models = optimized_hp_dataframe
                continue

            self._best_models = pd.concat([self._best_models, optimized_hp_dataframe])


    def get_all_best_models(self, metric_key: str = 'F1-score') -> pd.DataFrame:
        """
        Get all best models from all folds.

        :return best_models_per_fold: The best models from all folds.
        """
        if len(self._best_models)==0:
             raise ValueError("No models found for selection. ")
        self._sort_by(metric_key)
        return self._best_models.copy()


    def get_best_overall_model(self, metric_key: str = 'F1-score') -> pd.Series:
        """
        Selects the single best model.

        :return pd.Series object: Best model with its hyperparameters and
        metrics results.
        """

        best = self.get_all_best_models(metric_key)
        return best.iloc[0]


    def _sort_by(self, metric: str = 'F1-score') -> None:
        self._best_models.sort_values(by=[metric], ascending=False, inplace=True)
