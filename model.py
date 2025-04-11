import torch
from torch import nn
import numpy as np

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        super().__init__()

        self.linear1 = nn.Linear(input_dim,  2*output_dim)

        nn.init.orthogonal_(self.linear1.weight, gain=1.0)

        self.output_dim = output_dim

        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.linear1(x)

        print(x)

        distribution = torch.distributions.Normal(
            x[:self.output_dim],
            torch.exp(x[self.output_dim:] + 1e-6)
        )

        out = distribution.rsample()

        return out
    
    def getDistributionData(self, x, y):
        x = self.linear1(x)

        distribution = torch.distributions.Normal(
            x[:self.output_dim],
            torch.exp(x[self.output_dim:] + 1e-6)
        )

        log_prob = distribution.log_prob(y)
        entropy = distribution.entropy().sum()

        return log_prob, entropy

    def train(self, records):
        self.optim.zero_grad()
        for state, out, loss in records:
            # print(loss)
            log_prob, entropy = self.getDistributionData(state, out)
            policyLoss = loss * log_prob.sum() / len(records)
            # print(entropy)
            # policyLoss = 1 * entropy
            # print(policyLoss)
            policyLoss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

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

        print()

    def getLoss(self, state, out):
        # print(state, out, state + out)
        return -torch.norm(torch.clamp(state + out, -20, 20))