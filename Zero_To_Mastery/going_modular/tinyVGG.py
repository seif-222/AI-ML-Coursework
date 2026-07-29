import torch
from torch import nn
class VGGmodel (nn.Module):
    def __init__(self,
                 input_layers,
                 hidden_layers,
                 output_layers):
        super(VGGmodel, self).__init__()
        self.B1 = nn.Sequential(
            nn.Conv2d(in_channels = input_layers, out_channels = hidden_layers, kernel_size = 3,stride=1, padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_layers, out_channels = hidden_layers, kernel_size = 3,stride=1, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                     stride=2)
        )
        self.B2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_layers, out_channels = hidden_layers, kernel_size = 3,stride=1, padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_layers, out_channels = hidden_layers, kernel_size = 3,stride=1, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                     stride=2)
        )

        self.classifier =nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_layers*13*13
                      , out_features = output_layers),
        )

    def forward (self,x):
        return self.classifier(self.B2(self.B1(x)))
