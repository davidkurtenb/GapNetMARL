#HyperNetwork for generating target network weights.

import math

import torch
import torch.nn as nn


class HyperNetwork(nn.Module):

    def __init__(
        self,
        cond_dim,
        target_in_dim,
        target_out_dim,
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=False,
    ):
        super().__init__()
        self.target_in_dim = target_in_dim
        self.target_out_dim = target_out_dim

        num_weights = target_in_dim * target_out_dim

        #Weight MLP
        w_layers = []
        in_dim = cond_dim
        for _ in range(num_layers - 1):
            w_layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                w_layers.append(nn.LayerNorm(hidden_dim))
            w_layers.append(nn.ReLU())
            in_dim = hidden_dim
        w_layers.append(nn.Linear(in_dim, num_weights))
        self.weight_net = nn.Sequential(*w_layers)

        #Bias MLP
        b_layers = []
        in_dim = cond_dim
        for _ in range(num_layers - 1):
            b_layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                b_layers.append(nn.LayerNorm(hidden_dim))
            b_layers.append(nn.ReLU())
            in_dim = hidden_dim
        b_layers.append(nn.Linear(in_dim, target_out_dim))
        self.bias_net = nn.Sequential(*b_layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, conditioning):
        batch_size = conditioning.shape[0]
        weights = self.weight_net(conditioning)
        weights = weights.view(batch_size, self.target_in_dim, self.target_out_dim)
        biases = self.bias_net(conditioning)
        return weights, biases


def hyper_forward(embedding, weights, biases):
    # (batch, 1, in_dim) @ (batch, in_dim, out_dim) -> (batch, 1, out_dim) -> (batch, out_dim)
    output = torch.bmm(embedding.unsqueeze(1), weights).squeeze(1) + biases
    return output
