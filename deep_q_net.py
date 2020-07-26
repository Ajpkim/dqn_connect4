import torch
import torch.nn as nn
import torch.nn.functional as F

# ONE HOT THE STATE???  -> 1x84 input dim

# arch2: alphazero type arch
class DeepQNet3(nn.Module):
    def __init__(self):
        super(DeepQNet3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=126, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(126)

        self.res1

    def foward(self, board):
        x = self.encode_board(board)




    
    def encode_board(self, board):
        encoded_board = np.zeros((3,6,7))
        for row in range(6):
            for col in range(7):
                if board[row, col] == 0: encoded_board[0, row, col] = 1
                elif board[row, col] == 1: encoded_board[1, row, col] = 1
                else: encoded_board[2, row, col] = 1
        
        return encoded_board


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        pass


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        pass






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

