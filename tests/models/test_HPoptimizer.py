import os
import unittest
from typing import override

import torch

from recaptcha_classifier.models.main_model.HPoptimizer import HPOptimizer
from recaptcha_classifier.train.training import Trainer
from tests.models.utils_training_hpo import initialize_dummy_components


class TestHPOptimizer(unittest.TestCase):
    """Testing class for HPOptimizer"""

    @override
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model, train_loader, val_loader = initialize_dummy_components(8, 2, 2, 3)
        self.model = model
        trainer = Trainer(train_loader=train_loader,
                               val_loader=val_loader,
                               epochs=1,
                               save_folder='test_training_checkpoints')
        self.hpo = HPOptimizer(trainer)
        self.hp = [list(range(1,3)), list(range(3,5)), [1e-2]]
        self.hp_combos = self.hpo._generate_hp_combinations(self.hp)


    def test_generating_combos(self):
        true_combos = [(1,3,1e-2),
                      (1,4,1e-2),
                      (2,3,1e-2),
                      (2,4,1e-2)]
        self.assertEqual(true_combos, self.hp_combos,
                         msg="HP combinations don't match. "
                             f"{true_combos} not the same as {self.hp_combos}")

    def test_train_one_model(self):
        # MainModel is currently hardcoded
        self.hpo._train_one_model(self.hp_combos[0], save_checkpoints=True)
        assert os.path.exists(self.hpo.trainer.save_folder)


    def test_retrieve_results(self):
        results = self.hpo.optimize_hyperparameters(*self.hp, save_checkpoints=False)
        self.hpo.trainer.delete_checkpoints()
        print(results.head())
        self.assertEqual(len(results['loss']), 4)


    def test_recurring_hpo(self):
        results1 = self.hpo.optimize_hyperparameters(*self.hp, save_checkpoints=False)
        results2 = self.hpo.optimize_hyperparameters(*self.hp, save_checkpoints=False)
        print(results1.head())
        print(results2.head())
        self.hpo.trainer.delete_checkpoints()
        self.assertNotEqual(results1['loss'][0], results2['loss'][0])

if __name__ == '__main__':
    unittest.main()
