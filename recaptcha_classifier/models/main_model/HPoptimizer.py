

class HpOptimizer(object):
    """ ..."""

    def __init__(self, model):
        self.model = model

    def optimize_hyperparameters(self,
                                 n_layers: list = [1, 5],
                                 kernel_sizes: list = [3, 5],
                                 learning_rates: list = [1e-3, 1e-4]):
        """
        Main loop for optimizing hyperparameters.
        :param n_layers: list of integers specifying the number of hidden layers range.
        :param kernel_sizes: list of integers specifying the kernel sizes range.
        :param learning_rates: list of floats specifying the learning rate range.
        :return: model architecture performances ranked from best to worst.
        """

