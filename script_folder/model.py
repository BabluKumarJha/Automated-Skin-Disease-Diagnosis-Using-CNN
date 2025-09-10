# import important library and local script file
import torch
import torch.nn as nn
from data import * # import all from data file.

# device agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Creating cnn model.
class CNNModel(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int)->None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    # forward function.
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.global_pool(x)          # (batch, hidden_units, 1, 1)
        x = torch.flatten(x, 1)          # (batch, hidden_units)
        x = self.classifier(x)
        return x

cnn_model = CNNModel(input_shape = 3,
                     output_shape = len(class_names), # number of classes as we have 9 class in this problem.
                     hidden_units = 32).to(device)
# print(cnn_model)
# print(f"Model is on: {next(cnn_model.parameters()).device} device")

