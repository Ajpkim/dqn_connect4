import numpy as np

from connect4 import Connect4


class Connect4MDP(Connect4):
    def __init__(self, rows=6, cols=7, rewards={'win': 1, 'loss': -1, 'tie': 0}, discount_factor=0.95):
        super().__init__(rows, cols)
        self.rewards = rewards
        self.discount_factor = discount_factor
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

    def get_state(self):
        "update and return current state"
        self.state = self.board.flatten()
        return self.state
    
    def invalid_moves(self):
        invalid_cols = []
        for col in range(self.cols):
            if self.board[0, col] != 0:
                invalid_cols.append(col)
        return invalid_cols

    def get_flipped_state(self):
        state = self.get_state()
        flipped_state = np.zeros(len(state))
        for idx, element in enumerate(state):
            if element == 0:
                new = 0
            elif element == 1:
                new = -1
            else: new = 1 
            flipped_state[idx] = new
        return flipped_state
    
    def get_flipped_board(self):
        flipped_state = self.get_flipped_state()
        return flipped_state.reshape(self.rows, self.cols)

    def check_game_over(self):
        if self.status == 0:
            return False
        return True

if __name__ == '__main__':
    pass