import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, dueling, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dueling = dueling

        self.fc1 = nn.Linear(state_size, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 64)

        if self.dueling:
            self.fc_v = nn.Linear(64, 1, bias=True)
            self.fc_a = nn.Linear(64, action_size, bias=True)
        else:
            self.fc_final = nn.Linear(64, action_size, bias=True)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        if self.dueling:
            v = self.fc_v(x)
            a = self.fc_a(x)
            x = v + a - a.mean()
        else:
            x = self.fc_final(x)

        return x
        