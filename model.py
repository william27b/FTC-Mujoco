import torch
from torch import nn
import numpy as np

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=1):
        super().__init__()

        self.linear1 = nn.Linear(input_dim,  output_dim, bias=False)
        self.std = 1e-1

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
        self.optim.zero_grad()
        totalLoss = 0.0
        for state, out, loss in records:
            # print(loss)
            log_prob = self.getDistributionData(state, out)
            # print(loss, log_prob.sum())
            policyLoss = loss * log_prob.sum()
            # policyLoss = loss * log_prob.sum() / len(records)
            # policyLoss = policyLoss + 0.01 * std.sum()
            totalLoss += policyLoss
            # print(entropy)
            # policyLoss = 1 * entropy
            # print(policyLoss)
            policyLoss.backward()

        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad norm = {param.grad.norm().item()}")
            else:
                print(f"{name}: NO GRAD")

            if param.data is not None:
                print(f"{name}: data = {param.data}")
            else:
                print(f"{name}: NO VALUES")

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

        print("avg loss", totalLoss / len(records))
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
        return torch.sum((state[:2] * 3 * 144 - out) ** 2) + 20000 * ( -state[2] * out[1] +state[3] * out[0] +state[4] * out[1] -state[5] * out[0] )