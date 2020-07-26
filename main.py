import argparse
import logging
import os
import pdb
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml

from agent import Agent
from connect4 import Connect4
from deep_q_agent import DeepQAgent
from evaluator import Evaluator
from logger import setup_logger
from mdp import Connect4MDP
from replay_buffer import ReplayBuffer
from self_play_episodes import self_play_episodes
from trainer import Trainer
from util_players import RandomPlayer, HumanPlayer

## Try CNN and use 2d board state as input
# so One Hot the board for p1, p2 posistion. Only need 2dim since 2 blanks == empty space. 
# Q: would it help network learn if i redundantly give it 3 channels?
# One Hotting seems necessary regardless of cnn or not now
## seems best to handle input processing within the specific architectures and just pass board state around
# can handle board encoding as initial step in the forward pass and then just always give agent board state. Simpler.
    # just change self play epsiodes and a couple thigns in agent methods



# Should I do something more systematic wrt epochs and data generation (self play) as 
# in train through all data in replay buffer before filling it again for next epoch etc. 
# and use torch dataloaders?


## exploit symmetry by adding double entries for each experience of a flipped board? 
# Will this make it easier to learn symmetry?


# should store data on losses and eval performance with agent info as in memory and learning iter count


### Is it hard to learn connect4 with current archetecture bc relationship btwn states and rewards is NOT SMOOTH???
## Can I make learning tractable with some search help???
# --> network arch gets more complicated if I want to enable MCTS as well, multiple network heads computing
# prob of action being highest value as well as likelihood of next move being winning move if replicate alphazero


### Pretty sure I'm messing up the learning step and gradient while gathering q vals and next q vals for q targets
## Does loss fn expect 2d tensor?


## Write min max to test against.


### AGENT ONLY LEARNS TO CONSECUTIVELY PLAY IN THE SAME COL... 

## Should I add param for trying out different net architectures in agent class?

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default='config/default.yaml', help='Configuration file location.')
parser.add_argument("--load_file", default='', help='file load model for training from.')
parser.add_argument("--save_file", default='models/new_model_in_training', help='file to training model to.')
parser.add_argument("--log_file", default='logs/new_model_logs.log', help='log file location. Corresponds with model being trained.')
ARGS = parser.parse_args()

with open(ARGS.config_file, mode='r') as f:
    config = yaml.safe_load(f)

random_seed = config['random_seed']
torch.manual_seed(random_seed)
np.random.seed(random_seed)

setup_logger(ARGS.log_file)
logger = logging.getLogger(__name__)

logger.info('\n\n----------   NEW LOG   ----------\n')
logger.info(f'config_file: {ARGS.config_file}')
logger.info(f'load_file: {ARGS.load_file}')
logger.info(f'save_file: {ARGS.save_file}')
logger.info(f'config params: {config}')

model_dir, model_name = os.path.split(ARGS.save_file)
memory_file = model_dir + '/memory'
agent = DeepQAgent(name=model_name)
logger.info(f'Device: {agent.device}')

if ARGS.load_file:
    agent.load_model(ARGS.load_file)
    logger.info('loaded model')
    agent.load_memory_learning_iters(memory_file)
    logger.info('loaded memory and learning iters count')

trainer = Trainer(agent=agent, **config['Trainer'])
evaluator = Evaluator()

while len(agent.replay_buffer.memory) < agent.replay_buffer.capacity:
    trainer.self_play(100)
    logger.info('Building memory before learning')

#####################  TRAINING  ###########################
for epoch in range(config['epochs']):
    logger.info(f'EPOCH {epoch}')
    trainer.train(iters=config['iters'], n_episodes=config['n_episodes'])
    
    ## EVALUATION PHASE
    if config['eval_best']:
        best_agent = DeepQAgent(name='best')
        best_agent.load_model(config['best_model'])
        results, percentages = evaluator.evaluate(agent_1=agent, agent_2=best_agent, n_episodes=config['eval_episodes'])
        logger.info(f"Performance vs best_model over {config['eval_episodes']} games: {percentages}")
        
        if percentages[agent.name] > percentages[best_agent.name]:
            logger.info('WON VS BEST_MODEL. Overwriting best_model')
            agent.save_model(config['best_model'])
        else:
            logger.info('LOSS VS BEST_MODEL')
    
    if config['eval_random']:
        random_agent = RandomPlayer()
        results, percentages = evaluator.evaluate(agent_1=agent, agent_2=random_agent, n_episodes=config['eval_episodes'])
        win_p, loss_p, tie_p = [results[k] / config['eval_episodes'] for k in results]
        logger.info(f"Performance vs random agent over {config['eval_episodes']} games: {percentages}")
    
    agent.save_model(ARGS.save_file)
    logger.info(f'saved model state_dict at {ARGS.save_file}')
    agent.save_memory_learning_iters(memory_file)
    logger.info(f'saved memory and learning iters count at {memory_file}')

logger.info(f'Total agent learning iters: {agent.learning_iters}')
logger.info('DONE')
################################################





