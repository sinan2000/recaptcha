import unittest
import torch
from torch.optim import RAdam
from torch.optim.lr_scheduler import StepLR
import tempfile
import shutil
import os
import uuid
import gc

from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.models.main_model.model_class import MainCNN
from recaptcha_classifier.models.main_model.HPoptimizer import HPOptimizer
from recaptcha_classifier.models.main_model.kfold_validation import KFoldValidation
from tests.models.utils_training_hpo import create_dummy_data


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
        self.dummy_data = create_dummy_data(num_classes=3, num_samples=30)
        loader = torch.utils.data.DataLoader(self.dummy_data, batch_size=4)

        # Dummy model
        model = MainCNN(n_layers=1, kernel_size=3)
        optimizer = RAdam(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        unique_folder = os.path.join(self.temp_dir, str(uuid.uuid4()))
        os.makedirs(unique_folder, exist_ok=True)   

        self.trainer = Trainer(
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=1,
            save_folder=unique_folder,
            device=self.device
        )
        # Initialize optimizer and CV validator
        self.hp_optimizer = HPOptimizer(self.trainer)
        self.kfold_validator = KFoldValidation(
            data=self.dummy_data,
            k_folds=3,
            hp_optimizer=self.hp_optimizer,
            device=self.device
        )

    def test_kfold_and_hpo_run(self) -> None:
        """
        Running cross validation and asserting if it returns the
        necessary items.
        """
        # Run cross-validation
        self.kfold_validator.run_cross_validation(top_n_models=2,
                                                  save_checkpoints=False)

        # Get results
        results = self.kfold_validator.get_all_best_models()

        # Check result structure
        self.assertEqual(len(results), 3, "Should return results for 3 folds")
        for fold_models in results:
            self.assertEqual(len(fold_models), 2, "Should return top 2 models per fold")
            for model, metrics in fold_models:
                self.assertTrue(hasattr(model, "forward"), "Model should be valid")
                self.assertIn("F1-score", metrics, "Metrics should include F1-score")

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
