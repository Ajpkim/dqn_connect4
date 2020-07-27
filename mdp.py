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
        return self.get_state()

    def invalid_moves(self):
        invalid_cols = []
        for col in range(self.cols):
            if self.board[0, col] != 0:
                invalid_cols.append(col)
        return invalid_cols

    def get_state(self):
        "update and return current state"
        # self.state = self.board.flatten()
        
        ## updated for CNN... could make more efficient by handling in make move, since 
        ## would know exactly where to update with knowledge of most recent move and player id...
        state = np.zeros((2,6,7))
        for row in range(6):
            for col in range(7):
                if self.board[row, col] == 0: continue
                elif self.board[row, col] == 1: state[0, row, col] = 1
                else: state[1, row, col] = 1
        ### HANDLING PYTORCH CONVERSION IN REPLAY BUFFER / AGENT GENERALLY
        # state = torch.tensor(state, dtype=torch.float32)
        self.state = state
        return state

    def get_flipped_state(self):
        # flipped_state = self.get_state().clone()
        # flipped_state = torch.flip(flipped_state, (0,))
        flipped_state = self.get_state().copy()[::-1]
        return flipped_state

    ### Pre CNN method
    # def get_flipped_state(self):
    #     state = self.get_state()
    #     flipped_state = np.zeros(len(state))
    #     for idx, element in enumerate(state):
    #         if element == 0:
    #             new = 0
    #         elif element == 1:
    #             new = -1
    #         else: new = 1 
    #         flipped_state[idx] = new
    #     return flipped_state
    
    
    def get_flipped_board(self):
        flipped_board = np.zeros((self.rows, self.cols))
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row, col] == 0: continue
                elif self.board[row, col] == 1: flipped_board[row, col] = 2
                else: flipped_board[row, col] = 1
        
        return flipped_board


        # pre CNN method below:
        # flipped_state = self.get_flipped_state()
        # return flipped_state.reshape(self.rows, self.cols)

    def check_game_over(self):
        if self.status == 0:
            return False
        return True

if __name__ == '__main__':
    pass