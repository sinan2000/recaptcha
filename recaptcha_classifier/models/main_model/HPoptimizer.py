import itertools
import random
import pandas as pd

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
                                 n_layers: list = [1, 2, 3],
                                 kernel_sizes: list = [3, 5],
                                 learning_rates: list = [1e-2, 1e-3, 1e-4],
                                 save_checkpoints: bool = True,
                                 n_models: int = 1,
                                 n_combos: int = 8,  # Number of random samples
                                 ) -> pd.DataFrame:
        """
        Main loop for optimizing hyperparameters using Random Search.
        History is cleared every time the method is called.
        :param n_models: number of best models to return.
        :param n_combos: number of randomly selected combinations to optimize.
        :param save_checkpoints: If True, trainer saves checkpoints after each epoch.
        :param n_layers: list of integers specifying the number of hidden layers range.
        :param kernel_sizes: list of integers specifying the kernel sizes range.
        :param learning_rates: list of floats specifying the learning rate range.
        :return: pd.DataFrame with model architecture performances ranked from best to worst.
        """

        if max(n_combos, n_models) > (len(n_layers) * len(kernel_sizes) * len(learning_rates)):
            raise ValueError('n_combos and n_models should be '
                             'equal to or less than the number of HP combinations.')

        hp = [n_layers, kernel_sizes, learning_rates]

        # generating HP combinations:
        hp_combos = self._generate_hp_combinations(hp)
        random.shuffle(hp_combos)
        hp_combos = hp_combos[:n_combos]

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
        df_opt_data.sort_values(by=['loss'], ascending=True, inplace=True, ignore_index=True)
        return df_opt_data.copy()[:n_models]


    def _train_one_model(self, hp_combo) -> None:
        model = MainCNN(n_layers=int(hp_combo[0]), kernel_size=int(hp_combo[1]))
        self.trainer.train(model=model, lr=hp_combo[2], load_checkpoint=False)


    def _generate_hp_combinations(self, hp) -> list:
        return list(itertools.product(*hp))

    def _clear_history(self) -> None:
        for key in self._opt_data.keys():
            self._opt_data[key] = []

    def get_best_hp(self) -> list:
        """
        Returns the best hyperparameters based on the loss.
        :return: list of best hyperparameters.
        """
        df_opt_data = self.get_history()
        if len(df_opt_data) == 0:
            return []
        row = df_opt_data.iloc[0]
        return [int(row['layers']), int(row['kernel_sizes']), float(row['lr'])]