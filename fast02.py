from torch import nn, optim
import torch


class FastNN(nn.Module):
    def __init__(self, s_size, a_size):
        super(FastNN, self).__init__()
        self.num_layers = 1
        self.hidden_size = 128

        self.l1 = nn.Linear(s_size  + a_size, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, s_size)
        self.tanh = nn.Tanh()
        self.lr = 1e-4
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def reset(self, s):
        pass

    def loss(self, pred, target):
        return torch.mean((pred - target) ** 2)

    def forward(self, SA):

        # x = torch.cat([s, a], -1).unsqueeze(0)

        x = self.tanh(self.l1(SA))
        x = self.tanh(self.l2(x))
        x = self.tanh(self.l3(x))
        return x