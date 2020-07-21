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


###########

    # def self_play(self, mdp, episodes):
    #     "Simulate games against self and store experiences in replay buffer"
    #     experiences = self_play_episodes(mdp, self, episodes)
    #     for state, action, reward, next_state, done in experiences:
    #         self.replay_buffer.push(state, action, reward, next_state, done)

    # def save_checkpoint(self, path, iteration, loss):
    #     torch.save({'iterations': iteration,
    #                'model_state_dict': self.net.state_dict,
    #                'optimizer_state_dict': self.optimizer.state_dict(),
    #                'loss': loss}, 
    #                path)

    # def load_checkpoint(self, path):
    #     checkpoint = torch.load(path)
    #     iteration = checkpoint['iteration']
    #     model_state_dict = checkpoint['model_state_dict']
    #     optimizer_state_dict = checkpoint['optimizer_state_dict']
    #     loss = checkpoint['loss']

#######


    # def learn(self):
    #     "Update net with batch of experiences randomly drawn from replay buffer"
        
    #     self.optimizer.zero_grad()
    #     batch = self.replay_buffer.sample(self.batch_size)
       
    #     states = torch.tensor([x.state for x in batch]).float().to(self.device)
    #     actions = [x.action for x in batch]
    #     rewards = torch.tensor([x.reward for x in batch]).float().to(self.device)
    #     next_states = torch.tensor([x.next_state for x in batch]).float().to(self.device)
    #     dones = [x.done for x in batch]  

    #     # UPDATE LATER TO use policy & target net 

    #     q_vals = self.net(states)[range(len(actions)), actions]  # get the q_vals wrt actions taken
    #     q_next_vals = self.net(next_states)
    #     q_next_vals[dones] = 0  # set future value of terminal states to 0
    #     q_targets = rewards + gamma * torch.max(q_next_vals, dim=1)[0]

    #     loss = self.loss_fn(q_targets, q_vals).to(self.device)
    #     loss.backward()
    #     self.optimizer.step()

    #     # decay epsilon
    #     self.eps = self.eps * self.eps_decay if self.eps > self.eps_min else self.eps_min
    #     self.learn_iter += 1

    # def train(self, iters, n_episodes):
    #     for i in range(iters):
    #         self.self_play(self.mdp, n_episodes)
    #         self.learn()