import unittest
import torch
from torch.optim import RAdam
from torch.optim.lr_scheduler import StepLR
import tempfile
import shutil
import os
import uuid
import gc
from torch.utils.data import Subset

from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.models.main_model.model_class import MainCNN
from recaptcha_classifier.models.main_model.HPoptimizer import HPOptimizer
from recaptcha_classifier.models.main_model.kfold_validation import KFoldValidation
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data loading
        loaders = get_real_dataloaders()
        train_loader = loaders['train']
        val_loader = loaders['val']
        train_dataset = train_loader.dataset
        subset_indices = list(range(min(100, len(train_dataset))))
        small_dataset = Subset(train_dataset, subset_indices)

        # Model
        model = MainCNN(n_layers=1, kernel_size=3, num_classes=12)
        optimizer = RAdam(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        unique_folder = os.path.join(self.temp_dir, str(uuid.uuid4()))
        os.makedirs(unique_folder, exist_ok=True)   

        self.trainer = Trainer(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=1,
            save_folder=unique_folder,
            device=self.device
        )
        # Initialize optimizer and CV validator
        self.hp_optimizer = HPOptimizer(self.trainer)
        self.folds = 2
        self.kfold_validator = KFoldValidation(
            data=small_dataset,
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
        self.assertEqual((self.folds*18, 8), results.shape,
                         f"Should return ({self.folds*18}, 8) shape results for {self.folds} folds")
        self.assertEqual((8,), best.shape, "Should return top 1 model row with (1,8) shape.")

    def tearDown(self):
        """
        Deleting files if neded.
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
