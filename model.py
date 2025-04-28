import torch
from torch import nn
import numpy as np

# torch.autograd.set_detect_anomaly(True)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate):
        super(Model, self).__init__()

        self.linear1 = nn.Linear(input_dim, output_dim, bias=False)
        # self.linear2 = nn.Linear(8,  output_dim, bias=False)

        # nn.init.orthogonal_(self.linear1.weight, gain=10.0)

        # with torch.no_grad():
            # self.linear1.weight[:2, :2] = torch.tensor(np.array([
                # [3*144.0, 0],
                # [0, 3*144.0]
            # ]))

        #     self.linear1.weight[:, :] = torch.tensor([[ 4.3200e+02,  0.0000e+00,  2.7868e+01,  6.3883e+01,  2.9606e+01,
        #  -8.0722e+01, -1.3449e+02,  7.4067e+00],
        # [ 0.0000e+00,  4.3200e+02,  6.3585e+01, -6.4337e+01, -8.2833e+01,
        #  -5.0707e+01, -1.5936e+01, -1.1275e+02]])

        # for name, param in self.named_parameters():
            # self.linearParam = param

        self.std = 1e-3

        self.output_dim = output_dim

        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.linear1(x)
        # x = nn.functional.relu(x)
        # x = self.linear2(x)

        distribution = torch.distributions.Normal(
            x,
            self.std
        )

        out = distribution.rsample()

        return out
    
    def getDistributionData(self, x, y):
        x = self.linear1(x)
        # x = nn.functional.relu(x)
        # x = self.linear2(x)

        distribution = torch.distributions.Normal(
            x,
            self.std
        )

        log_prob = distribution.log_prob(y)

        return log_prob

    def train(self, records):
        if len(records) == 0: return

        self.optim.zero_grad()
        for state, out, loss in records:
            
            log_prob = self.getDistributionData(state, out)

            policyLoss = -loss * log_prob.sum()
            policyLoss.backward()

        self.optim.step()

        for param in self.parameters():
            print(param)

        # print()
        print()

    def getLoss(self, state, out):
        """
            loss with known out
        """
        # target = state @ self.targetWeights + self.targetBias
        # return torch.sum((out - target) ** 2)

        """
            loss with known model
        """
        # return torch.sum((self.linear1.weight - self.targetWeights.T) ** 2) + torch.sum((self.linear1.bias - self.targetBias) ** 2)

        """
            loss
        """

        direct_movement = 10 * torch.sum((state[:2] - (out / (3 * 144))) ** 2)
        # direct_movement = 0.0

        # acceleration_penalty = 0.1 * torch.sum((torch.abs((state[6:]) - (out / 100)) - (acceptable_acceleration)) ** 2)

        # acceleration_penalty = 1 - torch.cosine_similarity(state[6:], out, -1)
        # acceleration_penalty = 20 * acceleration_penalty

        keeping_distance = (1.0 * state[2] * out[1]) + (1.0 * state[3] * out[0]) + (-1.0 * state[4] * out[1]) + (-1.0 * state[5] * out[0])
        keeping_distance = 50 * keeping_distance

        return direct_movement - keeping_distance