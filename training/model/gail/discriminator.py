import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class DiscriminatorNN(nn.Module):
    """独立的时序编码器 + 动作 MLP 判别器。"""
    def __init__(self, state_dim, action_dim, hidden_dim, dropout_in=0.2, dropout=0.2):
        super(DiscriminatorNN, self).__init__()
        layers = []
        input_dim =  state_dim + action_dim
        layers.append(nn.Dropout(dropout_in))
        for output_dim in hidden_dim:
            layers.append(spectral_norm(nn.Linear(input_dim, output_dim)))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        self.fc_latent = nn.Sequential(*layers)
        self.fc_out = spectral_norm(nn.Linear(hidden_dim[-1], 1))

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = self.fc_latent(x)
        return self.fc_out(x)
