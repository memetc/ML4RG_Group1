import torch.nn as nn
import torch
import pandas as pd

class CombinedModel(nn.Module):
    def __init__(self, xgb_model, cnn_model, hidden_size, column_names):
        super(CombinedModel, self).__init__()
        self.column_names = column_names
        self.xgb_model = xgb_model
        self.cnn_model = cnn_model
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x_tabular, x_sequence):
        x_tabular_df = pd.DataFrame(x_tabular.cpu().numpy(), columns=self.column_names)
        xgb_out = torch.tensor(self.xgb_model.predict(x_tabular_df), dtype=torch.float32).view(-1, 1).to(x_tabular.device)

        # xgb_out = torch.tensor(self.xgb_model.predict(x_tabular.cpu()), dtype=torch.float32).view(-1, 1).to(x_tabular.device)
        cnn_out = self.cnn_model(x_sequence)
        x = torch.cat((xgb_out, cnn_out), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# Define the CNN model structure
class SimpleCNN(nn.Module):
    def __init__(self, num_features, cnn_filters, hidden_size, seq_length, kernel_size=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_features, cnn_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()

        # Calculate the output sequence length after conv1 layer
        conv1_output_length = seq_length - kernel_size + 1
        self.conv1_output_size = cnn_filters * conv1_output_length

        self.fc1 = nn.Linear(self.conv1_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape from [batch_size, 203, 4] to [batch_size, 4, 203]
        x = self.conv1(x)  # Conv1d expects [batch_size, num_channels, sequence_length]
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x