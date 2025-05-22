import torch
import torch.nn as nn

class MainCNN(BaseModel): # should inherit BaseModel(nn.Module)

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
        self.n_layers = n_layers
        self.kernel_size = kernel_size

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

            self.classifier = nn.Sequential(
                nn.Linear(current_channels, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x



