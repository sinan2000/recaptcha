import numpy as np

from recaptcha_classifier.models.main_model.model_class import MultiHeadModel
from recaptcha_classifier.models.training import Trainer


class HPOptimizer(object):
    """ ..."""

    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.opt_data = dict()

    def optimize_hyperparameters(self,
                                 n_layers: list = [1, 5],
                                 kernel_sizes: list = [3, 5],
                                 learning_rates: list = [1e-3, 1e-4],
                                 ):
        """
        Main loop for optimizing hyperparameters.
        :param n_layers: list of integers specifying the number of hidden layers range.
        :param kernel_sizes: list of integers specifying the kernel sizes range.
        :param learning_rates: list of floats specifying the learning rate range.
        :return: model architecture performances ranked from best to worst.
        """

        n = self._check_arg_dims(n_layers, kernel_sizes, learning_rates)
        layers = np.arange(n_layers, n, dtype=int)
        kernels = np.arange(kernel_sizes, n, dtype=int)
        learning_rates = np.arange(learning_rates, n, dtype=float)

        hp = [layers, kernels,learning_rates]

        hp_combos = np.array(np.meshgrid(hp)).T.reshape(-1, len(hp))

        for i in range(len(hp_combos)):
            hp_combo = hp_combos[i]
            model = MultiHeadModel(n_layers=int(hp_combo[0]), kernel_size=int(hp_combo[1]))
            self.trainer.train(model=model, load_checkpoint=False)




                '''name = 'model_' + str(i)
                self.opt_data[name] = {
                    'n_layers': layers
                }
                self.opt_data[name].append()'''

