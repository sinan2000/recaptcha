import itertools
import random
import pandas as pd
from typing import List
from src.models.main_model.resnet_inspired_model_block import MainCNN
from src.train.training import Trainer


class HPOptimizer(object):
    """Class for optimizing hyperparameters."""

    def __init__(self, trainer: Trainer) -> None:
        """
        The constructor for the HPOptimizer class.

        Args:
            trainer: An instance of the Trainer class.

        Returns:
            None
        """
        self.trainer = trainer
        self._opt_data = {'Model index': [],
                          'layers': [],
                          'kernel_sizes': [],
                          'lr': [],
                          'loss': [],
                          'accuracy': []}

    def get_history(self) -> pd.DataFrame:
        """
        Returns the optimization history as a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the optimization history.
        """
        df_opt_data = pd.DataFrame(self._opt_data)
        if len(self._opt_data['loss']) > 0:
            df_opt_data.sort_values(by=['loss'], ascending=True, inplace=True)
        return df_opt_data.copy()

    def optimize_hyperparameters(self,
                                 n_layers: list = [2, 3, 4],
                                 kernel_sizes: list = [3, 5],
                                 learning_rates: list = [1e-2, 1e-3, 5e-3, 5e-4],
                                 save_checkpoints: bool = True,
                                 n_models: int = 1,
                                 n_combos: int = 12,  # Number of random samples
                                 ) -> pd.DataFrame:
        """
        Main loop for optimizing hyperparameters using Random Search.
        History is cleared every time the method is called.
        Args:
            n_models: number of best models to return.
            n_combos: number of randomly selected combinations to optimize.
            save_checkpoints: If True, trainer saves checkpoints after each
                epoch.
            n_layers: list of integers specifying the number of hidden
                layers range.
            kernel_sizes: list of integers specifying the kernel sizes range.
            learning_rates: list of floats specifying the learning rate range.

        Returns:
            pd.DataFrame: A DataFrame containing the optimization history.
        """

        if max(n_combos, n_models) > (len(n_layers) * len(
             kernel_sizes) * len(learning_rates)):
            raise ValueError('n_combos and n_models should be '
                             'equal to or less than the number '
                             'of HP combinations.')

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

            curr_architecture = [i, hp_combo[
                0], hp_combo[1], hp_combo[2], loss, accuracy]

            v = 0
            for key in self._opt_data.keys():
                self._opt_data[key].append(curr_architecture[v])
                v += 1

        df_opt_data = pd.DataFrame(self._opt_data)

        df_opt_data.sort_values(by=[
            'loss'], ascending=True, inplace=True, ignore_index=True)

        self._save_history(df_opt_data)
        if n_models < 1:
            raise ValueError('n_models should be greater than 0.')

        return df_opt_data.copy()[:n_models]

    def _train_one_model(self, hp_combo: List, save_checkpoints: bool) -> None:
        """
        Trains a single model with the given hyperparameters.

        Args:
            hp_combo: A list containing the hyperparameters for the model.
            save_checkpoints: If True, trainer saves checkpoints after each
                epoch.

        Returns:
            None
        """
        model = MainCNN(n_layers=int(hp_combo[0]), kernel_size=int(
            hp_combo[1]))
        self.trainer.train(model=model, lr=hp_combo[2], load_checkpoint=False,
                           save_checkpoint=save_checkpoints)

    def _generate_hp_combinations(self, hp: list) -> list:
        """Generates all possible combinations of hyperparameters.

        Args:
            hp: A list of lists containing the possible values for each
                hyperparameter.

        Returns:
            A list of all possible combinations of hyperparameters.
        """
        return list(itertools.product(*hp))

    def _clear_history(self) -> None:
        """Clears the optimization history."""
        for key in self._opt_data.keys():
            self._opt_data[key] = []

    def get_best_hp(self) -> list:
        """
        Returns the best hyperparameters based on the loss.

        Returns:
            list: A list containing the best hyperparameters.
        """
        df_opt_data = self.get_history()
        if len(df_opt_data) == 0:
            return []
        row = df_opt_data.iloc[0]
        return [int(row['layers']), int(row['kernel_sizes']), float(row['lr'])]

    def _save_history(self, history: pd.DataFrame) -> None:
        """
        Saves the history of hyperparameter optimization.
        :param history: DataFrame with the history of hyperparameter optimization.
        """
        history.to_csv('hp_optimization_history.csv', index=False)
