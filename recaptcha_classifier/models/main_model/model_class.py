import torch
import torch.nn as nn
from recaptcha_classifier.constants import INPUT_SHAPE
from recaptcha_classifier.models.base_model import BaseModel


class MainCNN(BaseModel):

    def __init__(self,
                 n_layers: int,
                 # n_heads: int,
                 kernel_size: int,
                 num_classes: int = 12,
                 input_shape: tuple = INPUT_SHAPE,
                 base_channels: int = 32
                 ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        self.input_shape = input_shape
        current_channels = self.input_shape[0]


        for layer_idx in range(n_layers):
            output_channels = base_channels * (2 ** layer_idx)

            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=current_channels, out_channels=output_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            current_channels = output_channels

        flattened_features = self._get_conv_output(self.input_shape)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=flattened_features, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def _get_conv_output(self, shape) -> int:
        """Dynamically calculate conv layers' output size
        :param shape: input shape
        :return: output shape
        """
        dummy_input = torch.zeros(1, *shape)  # batch_size=1
        with torch.no_grad():
            for layer in self.layers:
                dummy_input = layer(dummy_input)
        size = dummy_input.view(1, -1).size(1)
        return size


    def forward(self, x) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x  # ! return logits, torch.softmax(x, dim=1) if needed for uncertainty

