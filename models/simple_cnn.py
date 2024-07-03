import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple single-conv layer model
    Input: (species, stress_condition, base_sequence)
    Each one-hot encoded
    Output: predicted TPM value for the given input
    """

    def __init__(self, *kwargs) -> None:
        super().__init__()

        # hyperparameters
        species_size = kwargs["species_size"] if "species_size" in kwargs else 30
        stress_condition_size = (
            kwargs["stress_condition_size"] if "stress_condition_size" in kwargs else 12
        )
        hidden_size = kwargs["hidden_size"] if "hidden_size" in kwargs else 30
        cnn_filers = kwargs["cnn_filers"] if "cnn_filers" in kwargs else species_size

        # activation functions
        self.activation = kwargs["activation"] if "activation" in kwargs else nn.ReLU()

        # Input layers
        self.input_base = nn.Conv2d(1, hidden_size, kernel_size=4)
        self.input_species = nn.Linear(species_size, hidden_size)
        self.input_stress_condition = nn.Linear(stress_condition_size, hidden_size)

        self.global_average_pool = nn.AdaptiveAvgPool1d(1)  # to get fixed size tensor

        # Hidden layer
        self.hidden = nn.Linear(
            species_size + stress_condition_size + cnn_filers, hidden_size
        )

        # Output layer
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, input):
        x_species, x_stress_condition, x_base = input
        x_base = self.input_base(x_base)
        x_base = self.activation(x_base)
        x_base = x_base.squeeze()
        x_base = self.global_average_pool(x_base)
        x_base = x_base.squeeze()

        if len(x_base.shape) == 1:
            x_base = x_base.unsqueeze(0)
        x = torch.cat((x_species, x_stress_condition, x_base), 1)

        x = self.hidden(x)
        x = self.activation(x)

        x = self.output(x)
        x = self.activation(x)
        x = x.squeeze()
        return x
