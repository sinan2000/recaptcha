import itertools

import pandas as pd
import torch

from recaptcha_classifier.models.main_model.model_class import MainCNN
from recaptcha_classifier.train.training import Trainer


class HPOptimizer(object):
    """Class for optimizing hyperparameters."""

    def __init__(self, trainer: Trainer):
        self._trainer = trainer
        self._opt_data = {'Models': [],
                         'hp combination': [],
                         'loss': [],
                         'accuracy': []}


    def optimize_hyperparameters(self,
                                 n_layers: list = list(range(1,6)),
                                 kernel_sizes: list = list(range(3,6)),
                                 learning_rates: list = [1e-2, 1e-3, 1e-4]
                                 ) -> pd.DataFrame:
        """
        Main loop for optimizing hyperparameters.
        :param n_layers: list of integers specifying the number of hidden layers range.
        :param kernel_sizes: list of integers specifying the kernel sizes range.
        :param learning_rates: list of floats specifying the learning rate range.
        :return: pd.DataFrame with model architecture performances ranked from best to worst.
        """

        hp = [n_layers, kernel_sizes, learning_rates]

        # generating HP combinations:
        hp_combos = self._generate_hp_combinations(hp)

        for i in range(len(hp_combos)):
            hp_combo = hp_combos[i]
            self._opt_data[f'model_{i}'] = list()
            self._train_one_model(hp_combo)

            final_train_history = self._trainer.loss_acc_history[-1]
            loss = final_train_history[0]
            accuracy = final_train_history[1]

            self._opt_data['Models'].append(i)
            self._opt_data['hp combination'].append(hp_combo)
            self._opt_data['loss'].append(loss)
            self._opt_data['accuracy'].append(accuracy)

        self._opt_data = pd.DataFrame(self._opt_data)
        self._opt_data.sort_values(by=['loss'], ascending=True, inplace=True)
        return self._opt_data.copy()


    def _train_one_model(self, hp_combo) -> None:
        model = MainCNN(n_layers=int(hp_combo[0]), kernel_size=int(hp_combo[1]))
        self._trainer.optimizer = torch.optim.RAdam(model.parameters(), lr=hp_combo[2])
        self._trainer.train(model=model, load_checkpoint=False)


    def _generate_hp_combinations(self, hp) -> list:
        return list(itertools.product(*hp))
