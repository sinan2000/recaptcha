import torch
import torch.nn as nn

class MultiHeadModel(): # should inherit BaseModel(nn.Module)

    def __init__(self,
                 n_layers: int,
                 # n_heads: int,
                 kernel_size: int,
                 num_classes: int = 3,
                 input_shape: tuple = (3, 224, 224),
                 base_channels: int = 32
                 ) -> None:
        super().__init__()

        # self.check_args(n_layers, kernel_size, num_classes, input_shape, base_channels)

        self.layers = nn.ModuleList()
        current_channels = input_shape[0]

        for layer_idx in range(n_layers):
            output_channels = base_channels * (2 ** layer_idx)

            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=current_channels, out_channels=output_channels,
                              kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            current_channels = output_channels

            self.flattened_features = self._get_conv_output(input_shape)

            self.classifier = nn.Sequential(
                nn.Linear(self.flattened_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

    def _get_conv_output(self, shape):
        """Dynamically calculate conv layers' output size"""
        dummy_input = torch.zeros(1, *shape)  # batch_size=1
        with torch.no_grad():
            for layer in self.layers:
                dummy_input = layer(dummy_input)
        return int(torch.prod(torch.tensor(dummy_input.shape[1:])))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x



