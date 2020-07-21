import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent import Agent
from deep_q_net import DeepQNet
from mdp import Connect4MDP
from replay_buffer import ReplayBuffer


class DeepQAgent(Agent):
    """
    Agent with deep nn for valuing game states and interacting with mdp envirment. 
    Methods for analyzing and interacting with mdp state.
    """
    def __init__(self, name='DQAgent', mem_size=10000):
        # super().__init__()  # don't really need to call super bc there is nothing in Agent class
        self.name = name
        self.replay_buffer = ReplayBuffer(capacity=mem_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = DeepQNet().to(self.device)
        self.learning_iters = 0
        ### ADD POLICY & TARGET NETS

    def select_action(self, mdp, eps):
        "Episilon greedy action selection given mdp"
        if random.random() < eps:
            action = random.choice(mdp.valid_moves())
        else:
            state = mdp.get_state()
            invalid_moves = mdp.invalid_moves()
            action_estimates = self.action_estimates(state)
            action_estimates[invalid_moves] = -float('inf')
            action = torch.argmax(action_estimates).item()
        return action 
    
    def get_next_move(self, board):
        "Return best valid move. For competition  with Luer API"
        invalid_moves = []
        for col in range(len(board[0])):
            if board[0, col] != 0:
                invalid_moves.append(col)
    
        state = self.encode_board(board)
        action_estimates = self.action_estimates(state)
        action_estimates[invalid_moves] = -float('inf')
        return torch.argmax(action_estimates).item()

    def action_estimates(self, state):
        "Return value estimates for all actions given state"
        state = torch.from_numpy(state).to(self.device).float()
        with torch.no_grad():
            action_estimates = self.net(state)
        return action_estimates
    

    def encode_board(self, board):
        return board.flatten()

    # def save_agent(self, path):
    #     with()


    def save_model(self, path):
        torch.save(self.net.state_dict(), path)
    
    def load_model(self, path):
        self.net.load_state_dict(torch.load(path))

    def __repr__(self):
        return f'Deep Q Agent: {self.name}'    

if __name__ == '__main__':
    pass

