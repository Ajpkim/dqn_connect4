import copy
import numpy as np
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


from agent import Agent
from deep_q_net import DeepQNet, DeepQNet2, DeepQNet3
from logger import *
from mdp import Connect4MDP
from replay_buffer import ReplayBuffer

log_file='testing_logs/AE.log'
logger = get_logger(__name__, log_file=log_file, level=10)

class DeepQAgent(Agent):
    """
    Agent with policy and target networks for approximating and learning Q values of 
    (state, action) pairs. Methods for analyzing and interacting with mdp state.
    
    Args:
        - mem_size: capacity of agent's replay buffer for storing recent experience tuples
        - target_freq: number of learning steps between updates to target esimtation network
    """
    def __init__(self, name='DQAgent', mem_size=10000):
        self.name = name
        self.action_space = [x for x in range(7)]
        self.replay_buffer = ReplayBuffer(capacity=mem_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DeepQNet3().to(self.device)  # CHECK ARCHITECTURE
        self.target_net = copy.deepcopy(self.policy_net)
        self.learning_iters = 0

    def select_action(self, state, eps, valid_moves):
        "Episilon greedy action selection for training"
        if random.random() < eps:
            action = random.choice(valid_moves)
        else:
            invalid_moves = [a for a in self.action_space if a not in valid_moves]
            action_estimates = self.action_estimates(state)
            action_estimates[invalid_moves] = -float('inf')
            action = torch.argmax(action_estimates).item()
        return action 
    
    def get_next_move(self, board):
        "Return best valid move"
        invalid_moves = []
        for col in range(len(board[0])):
            if board[0, col] != 0:
                invalid_moves.append(col)
    
        state = self.encode_board(board)
        action_estimates = self.action_estimates(state)
        action_estimates[invalid_moves] = -float('inf')
        return torch.argmax(action_estimates).item()

    def action_estimates(self, state):
        "Return Q value estimates for all actions given state"
        state = torch.from_numpy(state).to(self.device).float()
        with torch.no_grad():
            action_estimates = self.policy_net(state)
        
        
        # logger.info(f'state \n\n {state.reshape(6,7)}')
        # logger.info(f'action_estimates: {action_estimates}')
    

        return action_estimates

    def update_target_net(self):
        self.target_net = copy.deepcopy(self.policy_net) 
    
    def encode_board(self, board):
        ## ADJUSTING FOR CNN (WAS PREV JUST NP.FLATTEN(BOARD))
        # return board.flatten()
        "One-Hot the board for self, opponent. Return 2x6x7 array."
        encoded_board = np.zeros((2,6,7))
        for row in range(6):
            for col in range(7):
                if board[row, col] == 0: continue
                elif board[row, col] == 1: encoded_board[0, row, col] = 1
                else: encoded_board[1, row, col] = 1
        
        return encoded_board
        # return torch.tensor(encoded_board, dtype=torch.float32)

    def save_memory_learning_iters(self, path):
        with open(path, 'wb') as f:
            data = {'learning_iters': self.learning_iters, 'memory': self.replay_buffer.memory}
            pickle.dump(data, f)
    
    def load_memory_learning_iters(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            learning_iters, memory = data['learning_iters'], data['memory']
            self.learning_iters = learning_iters
            for state, action, reward, next_state, done in memory:
                self.replay_buffer.push(state, action, reward, next_state, done)
        
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def __repr__(self):
        return f'Deep Q Agent: {self.name}'    

if __name__ == '__main__':
    pass

