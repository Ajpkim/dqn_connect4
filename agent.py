class Agent:
    def __init__(self, id=1, name='nameme'):
        self.id = id
        self.name = name

    def greedy(self, state):
        "Return the estimated best move"
        raise NotImplementedError()

    def get_val(self, state, action):
        "Return estimated Q value of taking given action in state"
        raise NotImplementedError()

    def get_next_move(self, state):
        "Return next game move. Must be legal action."
        raise NotImplementedError()

    def __repr__(self):
        return f'Agent name: {self.name}'

