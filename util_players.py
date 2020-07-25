import random

class RandomPlayer:
    def __init__(self, name='random player'):
        self.name = name

    def valid_moves(self, board):
        valid_cols = []
        for i in range(board.shape[1]):
            if 0 in board[:, i]:
                valid_cols.append(i)
        return valid_cols

    def get_next_move(self, board):
        valid_moves = self.valid_moves(board)
        return random.choice(valid_moves)    
    
    def __len__(self):
        return 1

    def __repr__(self):
        return 'Random Connect4 Player'

class HumanPlayer:
    def __init__(self, name='human'):
        self.name = name

    def get_next_move(self, board):
        return int(input('select a move human: '))

    def __len__(self):
        return 1
    
    def __repr__(self):
        return 'Human Connect4 Player'
