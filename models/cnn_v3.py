import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNV3(nn.Module):
    """
    Simple single-conv layer model
    Input: (species, stress_condition, base_sequence)
    Each one-hot encoded
    Output: predicted TPM value for the given input
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

        # hyperparameters
        species_size = kwargs["species_size"] if "species_size" in kwargs else 30
        stress_condition_size = (
            kwargs["stress_condition_size"] if "stress_condition_size" in kwargs else 12
        )
        hidden_size = kwargs["hidden_size"] if "hidden_size" in kwargs else 30
        cnn_filters = kwargs["cnn_filters"] if "cnn_filters" in kwargs else hidden_size
        conv_kernel_size = kwargs["kernel_size"] if "kernel_size" in kwargs else 4

        # activation functions
        self.activation = kwargs["activation"] if "activation" in kwargs else nn.ReLU()

        # Input layers
        self.input_base = nn.Conv1d(4, cnn_filters, kernel_size=conv_kernel_size)
        self.bn_base = nn.BatchNorm1d(cnn_filters)  # Batch normalization for conv layer

        self.input_species = nn.Linear(species_size, hidden_size)
        self.bn_species = nn.BatchNorm1d(hidden_size)  # Batch normalization for species

        self.input_stress_condition = nn.Linear(stress_condition_size, hidden_size)
        self.bn_stress_condition = nn.BatchNorm1d(
            hidden_size
        )  # Batch normalization for stress condition

        self.global_average_pool = nn.AdaptiveAvgPool1d(1)  # to get fixed size tensor

        # Hidden layer
        
        self.hidden = nn.Linear(hidden_size + hidden_size + cnn_filters, hidden_size)
        self.bn_hidden = nn.BatchNorm1d(
            hidden_size
        )  # Batch normalization for hidden layer

        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.bn_hidden2 = nn.BatchNorm1d(
            hidden_size
        )

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Output layer
        self.output = nn.Linear(hidden_size, 1)

        # Apply Xavier Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input):
        x_species, x_stress_condition, x_base = input

        # Base sequence processing
        x_base = x_base.permute(0, 3, 2, 1).squeeze()  
        x_base = self.input_base(x_base)
        x_base = self.bn_base(x_base)  # Apply batch normalization
        x_base = self.activation(x_base)
        x_base = x_base.squeeze()
        x_base = self.global_average_pool(x_base)
        x_base = x_base.squeeze()

        # Process species and stress condition
        x_species = self.input_species(x_species)
        x_species = self.bn_species(x_species)  # Apply batch normalization
        x_species = self.activation(x_species)

        x_stress_condition = self.input_stress_condition(x_stress_condition)
        x_stress_condition = self.bn_stress_condition(
            x_stress_condition
        )  # Apply batch normalization
        x_stress_condition = self.activation(x_stress_condition)

        # Concatenation
        x = torch.cat((x_species, x_stress_condition, x_base), dim=1)

        # Hidden layer
        x = self.hidden(x)
        x = self.bn_hidden(x)  # Apply batch normalization
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = self.bn_hidden2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Output layer
        x = self.output(x)
        x = x.squeeze()

        return x
