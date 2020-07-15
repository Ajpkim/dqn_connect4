import numpy as np


from DeepQAgent import DeepQAgent
from self_play_episodes import self_play_episodes
from train import train
from util_players import RandomPlayer, HumanPlayer
from agent import Agent
from Connect4 import Connect4
from mdp import Connect4MDP
from replay_buffer import ReplayBuffer



id = 1
name = 'test agent'
action_space = [0, 1, 2, 3, 4, 5, 6]
lr = .001
gamma = .9
batch_size = 64
mem_size = 10000
eps = 1
eps_min = .01
eps_decay = .9999
agent = DeepQAgent(id, name, action_space, lr, gamma,
                   batch_size, mem_size, eps, eps_min, eps_decay)

mdp = Connect4MDP()
# experiences = self_play_episodes(mdp, agent, 1)
# train(agent, iters=1, n_episodes=5)
