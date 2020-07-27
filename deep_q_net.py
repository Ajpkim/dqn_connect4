import torch
import torch.nn as nn
import torch.nn.functional as F



################################################################################################
### CNN Arch

# arch2: alphazero type arch
# See links below for general arch structure
# https://www.youtube.com/watch?v=OPgRNY3FaxA
# https://github.com/plkmo/AlphaZero_Connect4/blob/master/src/alpha_net_c4.py

# maybe max pooling isnt that helpful in case of approxing game state val vs image recognition type task

# out_channels = # of filters in layer  ('feature maps')

# for flattening from conv...:
# outputSizeOfCov = [(inputSize + 2*pad - filterSize)/stride] + 1

# conv filter weight are repr with single 4d tensor (output_channels, input_channels (depth), kernelX, kernelY)
# each filter has depth wrt input channels that are then convolved

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=84, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=84)

    def forward(self, x):
        x = x.view(-1, 2, 6, 7)  # flexible batch sizes
        x = F.relu(self.bn1(self.conv1(x)))
        return x


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=84, out_channels=84, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=84)
        self.conv2 = nn.Conv2d(in_channels=84, out_channels=84, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=84)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # add original input back in to output ('skip connection')
        x = F.relu(x)
        return x


class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=84, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)  # NO padding
        self.bn1 = nn.BatchNorm2d(num_features=2)
        self.fc1 = nn.Linear(in_features=2*6*7, out_features=42, bias=True)
        self.fc2 = nn.Linear(in_features=42, out_features=7, bias=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))        
        x = x.view(-1, 2*6*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeepQNet3(nn.Module):
    def __init__(self):
        super(DeepQNet3, self).__init__()
        self.conv = ConvBlock()
        for i in range(10):
            setattr(self, "res%i" % i, ResidualBlock())
        self.out = OutBlock() 
        
    def forward(self, x):
        x = self.conv(x)
        for res_layer in range(10):
            x = getattr(self, "res%i" % res_layer)(x)
        x = self.out(x)
        return x


    # def encode_boards(self, board_list):
    #         all_boards = []
    #         for board in board_list:
    #             encoded_board =  np.zeros((2,6,7))
    #             for row in range(6):
    #                 for col in range(7):
    #                     if board[row, col] == 0: continue
    #                     elif board[row, col] == 1: encoded_board[0, row, col] = 1
    #                     else: encoded_board[1, row, col] = 1
    #             encoded_board = torch.tensor(encoded_board, dtype=torch.float32)
    #             all_boards.append(encoded_board)
            
    #         return torch.stack(all_boards)

    

### WILL THIS CAUSE DEVICE ISSUES? DOES THIS NEED TO HANDLED IN AGENT CLASS AND SENT DIRECTLY TO DEVICE???
### YEAH, this will cause issues with minibatches and whatnot
    # def encode_board(self, board):
    #     encoded_board = np.zeros((2,6,7))
    #     for row in range(6):
    #         for col in range(7):
    #             if board[row, col] == 0: continue
    #             elif board[row, col] == 1: encoded_board[0, row, col] = 1
    #             else: encoded_board[1, row, col] = 1

    #     return torch.tensor(encoded_board, dtype=torch.float32)



################################################################################################
### Simple Feed Foward Archs

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

