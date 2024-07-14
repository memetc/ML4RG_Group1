import torch.nn as nn
import torch
import pandas as pd

class CombinedModel2(nn.Module):
    def __init__(self, xgb_model, cnn_model, hidden_size, column_names):
        super(CombinedModel2, self).__init__()
        self.column_names = column_names
        self.xgb_model = xgb_model
        self.cnn_model = cnn_model
        self.fc1 = nn.Linear(1 + hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x_tabular, x_sequence):
        x_tabular_df = pd.DataFrame(x_tabular.cpu().numpy(), columns=self.column_names)
        xgb_out = torch.tensor(self.xgb_model.predict(x_tabular_df), dtype=torch.float32).view(-1, 1).to(x_tabular.device)
        cnn_out = self.cnn_model(x_sequence)
        x = torch.cat((xgb_out, cnn_out), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x




