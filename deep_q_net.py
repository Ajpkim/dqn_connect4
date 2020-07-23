import torch
import torch.nn as nn
import torch.nn.functional as F

# ONE HOT THE STATE???  -> 1x84 input dim

## arch0
class DeepQNet(nn.Module):
    def __init__(self):
    # def __init__(self, input_dim, num_hidden_units, hidden_layers, output_dim):
        super(DeepQNet, self).__init__()
        self.fc1 = nn.Linear(42, 84)
        self.fc2 = nn.Linear(84, 84)
        self.fc3 = nn.Linear(84, 84)
        self.fc4 = nn.Linear(84, 24)
        self.fc5 = nn.Linear(24, 7)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x 

## arch1
class DeepQNet2(nn.Module):
    def __init__(self):
        super(DeepQNet2, self).__init__()
        self.fc1 = nn.Linear(42, 42)
        self.fc2 = nn.Linear(42, 24)
        self.fc3 = nn.Linear(24, 7)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 