import torch
from torch import nn
import numpy as np

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate):
        super().__init__()

        self.linear1 = nn.Linear(input_dim,  output_dim, bias=False)
        nn.init.orthogonal_(self.linear1.weight, gain=10.0)

        self.std = 1e-3

        self.output_dim = output_dim

        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.linear1(x)

        distribution = torch.distributions.Normal(
            x,
            self.std
        )

        out = distribution.rsample()

        return out
    
    def getDistributionData(self, x, y):
        x = self.linear1(x)

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

            policyLoss = loss * log_prob.sum() / len(records)
            policyLoss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=50.0)

        self.optim.step()

        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad norm = {param.grad.norm().item()}")
            else:
                print(f"{name}: NO GRAD")

            if param.data is not None:
                print(f"{name}: data = {param.data}")
            else:
                print(f"{name}: NO VALUES")

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

        #                                                            forward distance * y movement  right distance * x movement  backward distance * -y movement  left distance * -x movement
        return 100 * torch.sum((state[:2] - out / (3 * 144)) ** 2) + (state[2] * out[1] * 1.0)    + (state[3] * out[0] * 1.0)   + (state[4] * out[1] * -1.0)   +  (state[5] * out[0] * -1.0)