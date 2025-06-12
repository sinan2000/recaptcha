import unittest
import torch
from sklearn.model_selection import train_test_split
from torch.optim import RAdam
from torch.optim.lr_scheduler import StepLR
import tempfile
import shutil
import os
import uuid
import gc
from torch.utils.data import Subset, DataLoader

from src.train.training import Trainer
from src.models.main_model.model_class import MainCNN
from src.models.main_model.HPoptimizer import HPOptimizer
from src.models.main_model.kfold_validation import KFoldValidation
from tests.integration.get_real_data import get_real_dataloaders


class TestKFoldHPOIntegration(unittest.TestCase):
    """
    Integration test for cross validation + hp optimizer.
    """
    def setUp(self) -> None:
        """
        Setting up.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cuda" if torch.cuda
                                   .is_available() else "cpu")

        # Data loading
        loaders = get_real_dataloaders()
        val_loader = loaders['val']
        full_dataset = val_loader.dataset

        num_samples = len(full_dataset)
        all_indices = list(range(num_samples))

        train_indices, val_indices = train_test_split(
            all_indices, test_size=0.2, shuffle=True, random_state=42
        )

        # reducing the number of samples for speed (only 10% of each split)
        small_train_indices = train_indices[:int(0.1 * len(train_indices))]
        small_val_indices = val_indices[:int(0.1 * len(val_indices))]

        # subset datasets
        small_train_set = Subset(full_dataset, small_train_indices)
        small_val_set = Subset(full_dataset, small_val_indices)

        # real DataLoaders
        train_loader = DataLoader(small_train_set, batch_size=2, shuffle=True)
        val_loader = DataLoader(small_val_set, batch_size=2, shuffle=False)

        # Model
        model = MainCNN(n_layers=1, kernel_size=3, num_classes=12)
        optimizer = RAdam(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        unique_folder = os.path.join(self.temp_dir, str(uuid.uuid4()))
        os.makedirs(unique_folder, exist_ok=True)

        self.trainer = Trainer(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            save_folder=unique_folder,
            device=self.device
        )
        # Initialize optimizer and CV validator
        self.hp_optimizer = HPOptimizer(self.trainer)
        self.folds = 2
        self.kfold_validator = KFoldValidation(
            train_loader=train_loader,
            val_loader=val_loader,
            k_folds=self.folds,
            hp_optimizer=self.hp_optimizer,
            device=self.device
        )

    def test_kfold_and_hpo_run(self) -> None:
        """
        Running cross validation and asserting if it returns the
        necessary items.
        """
        # Run cross-validation
        self.kfold_validator.run_cross_validation(save_checkpoints=False)

        # Get results
        results = self.kfold_validator.get_all_best_models()
        best = self.kfold_validator.get_best_overall_model()

        # Check result structure
        self.assertEqual((self.folds, 8), results.shape,
                         f"Should return ({self.folds}, 8) shape results "
                         f"for {self.folds} folds")
        self.assertEqual((8,), best.shape, "Should return top 1 model row with "
                         "(1,8) shape.")

    def tearDown(self):
        """
        Deleting files if needed.
        """
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Warning during cleanup: {e}")

        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            import time
            time.sleep(0.5)
            shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
