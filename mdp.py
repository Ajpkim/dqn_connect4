import numpy as np
import torch

from connect4 import Connect4


class Connect4MDP(Connect4):
    def __init__(self, rows=6, cols=7, rewards={'win': 1, 'loss': -1, 'tie': 0}):
        super().__init__(rows, cols)
        self.rewards = rewards
        self.state = self.get_state()  

    def reset(self):
        self.board = np.zeros((self.rows, self.cols))
        self.status = 0
        self.turns = 0

    def reward_fn(self, id) -> int:
        if self.status == 0:
            return 0.0  # ongoing game

        if self.status == 'tie':
            return self.rewards['tie']

        if self.status == id:            
            return self.rewards['win']
        else:
            return self.rewards['loss']

    def make_move(self, col, id):
        super().make_move(col, id)

    def invalid_moves(self):
        invalid_cols = []
        for col in range(self.cols):
            if self.board[0, col] != 0:
                invalid_cols.append(col)
        return invalid_cols

    def get_state(self) -> np.array:
        "update and return current state as a 3x6x7 array"
        state = np.zeros((3,6,7))
        for row in range(6):
            for col in range(7):
                if self.board[row, col] == 0: continue
                elif self.board[row, col] == 1: state[0, row, col] = 1
                else: state[1, row, col] = 1

        # indicate turn
        if self.turns % 2 == 0: 
            state[2,::] = 1
        
        self.state = state
        return state

    def check_game_over(self):
        if self.status == 0:
            return False
        return True

if __name__ == '__main__':
    pass