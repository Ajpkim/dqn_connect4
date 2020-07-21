import numpy as np
import pdb

from datetime import datetime

from agent import Agent
from connect4 import Connect4
from deep_q_agent import DeepQAgent
from evaluator import Evaluator
from mdp import Connect4MDP
from replay_buffer import ReplayBuffer
from self_play_episodes import self_play_episodes
from trainer import Trainer
from util_players import RandomPlayer, HumanPlayer


### MDP
mdp = Connect4MDP()

###### Agent
id = 1
name = 'test agent'
mem_size = 10000
agent = DeepQAgent(id, name, mem_size)

#### Trainer
lr = .001
gamma = .9
batch_size = 64
eps = .1
iters = 100
n_episodes = 5
trainer = Trainer(agent, lr=.005,gamma=.9, iters=10000,n_episodes=5,batch_size=64,eps=.1)

######################################################

start = datetime.now()
start_time = start.strftime("%H:%M:%S")
print("Start time =", start_time); print()


trainer.self_play(50)
trainer.train()

##### Evaluator
evaluator = Evaluator()

################################################
rando = RandomPlayer()

res = evaluator.evaluate(agent, rando, 5000)
print(res)


#######
print()
end = datetime.now()
end_time = end.strftime("%H:%M:%S")
print("end time =", end_time)
running_time = end - start
print("running time =", running_time)
#######
