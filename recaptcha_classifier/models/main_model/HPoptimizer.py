import itertools

import numpy as np
import torch

from recaptcha_classifier.models.main_model.model_class import MainCNN
from recaptcha_classifier.train.training import Trainer


class HPOptimizer(object):
    """Class for optimizing hyperparameters."""

    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.opt_data = dict()

    def optimize_hyperparameters(self,
                                 n_layers: list = list(range(1,6)),
                                 kernel_sizes: list = list(range(3,6)),
                                 learning_rates: list = [1e-2, 1e-3, 1e-4]
                                 ):
        """
        Main loop for optimizing hyperparameters.
        :param n_layers: list of integers specifying the number of hidden layers range.
        :param kernel_sizes: list of integers specifying the kernel sizes range.
        :param learning_rates: list of floats specifying the learning rate range.
        :return: model architecture performances ranked from best to worst.
        """

        hp = [n_layers, kernel_sizes, learning_rates]

        # generating HP combinations:
        hp_combos = self._generate_hp_combinations(hp)

        for i in range(len(hp_combos)):
            hp_combo = hp_combos[i]
            self.opt_data[f'model_{i}'] = list()
            self._train_one_model(hp_combo)
            self.opt_data[f'model_{i}'] = [hp_combo, self.trainer.loss_acc_history[-1]]
            # SORT MODELS BY LOSS/ACC


    def _train_one_model(self, hp_combo) -> None:
        model = MainCNN(n_layers=int(hp_combo[0]), kernel_size=int(hp_combo[1]))
        self.trainer.optimizer = torch.optim.RAdam(model.parameters(), lr=hp_combo[2])
        self.trainer.train(model=model, load_checkpoint=False)


    def _generate_hp_combinations(self, hp) -> list:
        return list(itertools.product(*hp))
