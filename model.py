import torch
from torch import nn
import numpy as np
import math

# torch.autograd.set_detect_anomaly(True)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate):
        super(Model, self).__init__()

        # was kinda good
        # self.linear1 = nn.Linear(input_dim, output_dim, bias=True)

        self.d_model = input_dim
        self.num_heads = 1

        self.ffEncode = nn.Linear(input_dim, self.d_model)

        self.positionalEncoding = PositionalEncoding(d_model=input_dim, dropout=0.0, max_len=5000)
        
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads, dim_feedforward=2048, dropout=0.0)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer=self.encoderLayer, num_layers=2)

        self.actionHead = nn.Linear(self.d_model, output_dim)

        self.std = 1e-1

        self.output_dim = output_dim

        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # x = self.linear1(x)

        x = x.transpose(0, 1)
        x = self.ffEncode(x)
        x = self.positionalEncoding(x)
        x = self.transformerEncoder(x)
        x = self.actionHead(x)

        distribution = torch.distributions.Normal(
            x,
            self.std
        )

        return distribution

    def forwardAction(self, x):
        distribution = self.forward(x)
        return distribution.rsample()
    
    def forwardBackprop(self, x, y):
        distribution = self.forward(x)
        return distribution.log_prob(y)

    def train(self, records):
        if len(records) == 0: return

        self.optim.zero_grad()
        
        states = torch.Tensor()

        for index, (state, out, loss) in enumerate(records[2:]):
            states = torch.cat((states, state.view((1, 1, -1))))

            log_prob = self.forwardBackprop(states, out)

            policyLoss = -loss * log_prob.sum()
            policyLoss.backward()

        self.optim.step()

    def getLoss(self, state, out, collisionDirection):
        direct_movement = 200 * torch.sum((state[:2] - (out / (3 * 144))) ** 2)
        # direct_movement = 0.0

        # acceleration_penalty = 0.1 * torch.sum((torch.abs((state[6:]) - (out / 100)) - (acceptable_acceleration)) ** 2)

        # acceleration_penalty = 1 - torch.cosine_similarity(state[6:], out, -1)
        # acceleration_penalty = 20 * acceleration_penalty

        keeping_distance = (1.0 * state[2] * out[1]) + (1.0 * state[3] * out[0]) + (-1.0 * state[4] * out[1]) + (-1.0 * state[5] * out[0])
        keeping_distance = 25 * keeping_distance

        # collision_penalty = collisionDirection[0] * (out[0] / (3 * 144)) + collisionDirection[1] * (out[1] / (3 * 144))
        # collision_penalty *= 1000

        return -direct_movement# + keeping_distance# - collision_penalty