import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_features, cnn_filters, hidden_size, seq_length, kernel_size=3, stride=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_features, cnn_filters, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU()

        # Calculate the output sequence length after conv1 layer
        conv1_output_length = (seq_length - kernel_size) // stride + 1
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