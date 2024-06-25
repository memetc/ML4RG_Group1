import torch
import torch.nn as nn


class HierarchicalCNN(nn.Module):
    def __init__(self, **kwargs):
        super(HierarchicalCNN, self).__init__()

        # hyperparameters
        self.species_size = kwargs["species_size"] if "species_size" in kwargs else 30
        self.stress_condition_size = (
            kwargs["stress_condition_size"] if "stress_condition_size" in kwargs else 12
        )

        # activation functions
        self.activation = kwargs["activation"] if "activation" in kwargs else nn.ReLU()

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, input):
        species, stress_condition, base_sequence = input

        x = self.conv1(base_sequence)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)

        base_sequence_length = base_sequence.size(2)
        flatten_size = (base_sequence_length // 4) * 64
        x = x.view(-1, flatten_size)
        x = torch.cat((x, species, stress_condition), dim=1)

        fc1 = nn.Linear(
            flatten_size + self.species_size + self.stress_condition_size, 128
        )
        x = self.activation(fc1(x))
        x = self.activation(self.fc2(x))
        x = self.output(x)

        return x
