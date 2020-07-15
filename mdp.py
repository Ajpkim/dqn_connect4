class Connect4MDP(Connect4):
    def __init__(self, rows=6, cols=7, rewards={'win': 1, 'loss': -1, 'tie': 0}, discount_factor=0.95):
        super().__init__(rows, cols)
        self.rewards = rewards
        self.discount_factor = discount_factor
        # self.state  ## USE THIS FOR ENCODED STATE REPRESENTATION I.E. ONE HOTTING PIECES

    def reset(self):
        self.board = np.zeros((self.rows, self.cols))
        self.status = 0
        self.turns = 0

    def reward_fn(self, id) -> int:
        # ongoing game
        if self.status == 0:
            return 0.0

        # tie game
        if self.status == -1:
            return self.rewards['tie']

        # p1 win
        if self.status == 1:
            if id == 1:
                return self.rewards['win']
            else:
                return self.rewards['loss']

        # p2 win
        if id == 1:
            return self.rewards['loss']
        else:
            return self.rewards['win']

    def make_move(self, col, id):
        super().make_move(col, id)
        return self.board2vec()

    def invalid_moves(self):
        invalid_cols = []
        for col in range(self.cols):
            if self.board[0, col] != 0:
                invalid_cols.append(col)
        return invalid_cols

    def check_game_over(self):
        if self.state == 0:
            return False
        return True

    def board2vec(self):
        return self.board.flatten()

    # def board2state(self):
    #     flat = self.board2vec()
    #     state = 0
    #     for i in range(len(flat)):
    #         n = flat[i]
    #         state += n * 10**i
    #     return state

    def state2board(self):
        pass

    def state2vec(self):
        pass

    def vec2state(self):
        pass

    def print_state(self, state):
        print(self.state2board(state))
