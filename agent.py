class Agent:
    def __init__(self, id=1, name='nameme'):
        self.id = id
        self.name = name

    def best(self, state):
        "Return the estimated best move"
        raise NotImplementedError()

    def get_val(self, state, action):
        "Return estimated value of taking given action in state"
        raise NotImplementedError()

    def get_next_move(self, state):
        "Return next game move"
        raise NotImplementedError()

    def __repr__(self):
        return f'Agent name: {self.name}'

if __name__ == '__main__':
    pass