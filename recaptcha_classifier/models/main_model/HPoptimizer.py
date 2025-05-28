import itertools

import pandas as pd
import torch

from recaptcha_classifier.models.main_model.model_class import MainCNN
from recaptcha_classifier.train.training import Trainer


class HPOptimizer(object):
    """Class for optimizing hyperparameters."""

    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self._opt_data = {'Model index': [],
                         'layers': [],
                         'kernel_sizes': [],
                         'lr': [],
                         'loss': [],
                         'accuracy': []}


    def get_history(self)->pd.DataFrame:
        df_opt_data = pd.DataFrame(self._opt_data)
        if len(self._opt_data['loss']) > 0:
            df_opt_data.sort_values(by=['loss'], ascending=True, inplace=True)
        return df_opt_data.copy()


    def optimize_hyperparameters(self,
                                 n_layers: list = list(range(1,3)),
                                 kernel_sizes: list = [3, 5],
                                 learning_rates: list = [1e-3, 1e-4],
                                 save_checkpoints: bool = True
                                 ) -> pd.DataFrame:
        """
        Main loop for optimizing hyperparameters. History is cleared every time the method is called.
        :param save_checkpoints: If True, trainer saves checkpoints after each epoch.
        :param n_layers: list of integers specifying the number of hidden layers range.
        :param kernel_sizes: list of integers specifying the kernel sizes range.
        :param learning_rates: list of floats specifying the learning rate range.
        :return: pd.DataFrame with model architecture performances ranked from best to worst.
        """

        hp = [n_layers, kernel_sizes, learning_rates]

        # generating HP combinations:
        hp_combos = self._generate_hp_combinations(hp)

        if len(self._opt_data['loss']) != 0:
            self._clear_history()

        for i in range(len(hp_combos)):
            hp_combo = hp_combos[i]
            self._train_one_model(hp_combo, save_checkpoints)

            final_train_history = self.trainer.loss_acc_history[-1]
            loss = final_train_history[0]
            accuracy = final_train_history[1]

            curr_architecture = [i, hp_combo[0], hp_combo[1], hp_combo[2], loss, accuracy]

            v = 0
            for key in self._opt_data.keys():
                self._opt_data[key].append(curr_architecture[v])
                v+=1

        df_opt_data = pd.DataFrame(self._opt_data)
        df_opt_data.sort_values(by=['loss'], ascending=True, inplace=True)
        return df_opt_data.copy()


    def _train_one_model(self, hp_combo, save_checkpoints) -> None:
        model = MainCNN(n_layers=int(hp_combo[0]), kernel_size=int(hp_combo[1]))
        self.trainer.train(model=model, lr=hp_combo[2], load_checkpoint=False, save_checkpoint=save_checkpoints)


    def _generate_hp_combinations(self, hp) -> list:
        return list(itertools.product(*hp))

    def _clear_history(self) -> None:
        for key in self._opt_data.keys():
            self._opt_data[key] = []
