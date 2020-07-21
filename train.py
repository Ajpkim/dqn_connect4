import argparse
import logging
import os
import numpy as np
import pdb
import sys
import yaml

from pathlib import Path
from datetime import datetime

from agent import Agent
from connect4 import Connect4
from deep_q_agent import DeepQAgent
from evaluator import Evaluator
from logger import get_logger
from mdp import Connect4MDP
from replay_buffer import ReplayBuffer
from self_play_episodes import self_play_episodes
from trainer import Trainer



parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default='config/default.yaml', help='Configuration file path')
ARGS = parser.parse_args()

with open(ARGS.config_file, mode='r') as f:
    config = yaml.safe_load(f)

# HOW CAN I DO THIS W/O CALLING BASICCONFIG()?
logging.basicConfig(filename=config['files']['log_file'], 
                    filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

logger.info('\n--------------------------------\n'\
              '---------    NEW LOG   --------\n'\
              '--------------------------------\n')
logger.info(f'config_file: {ARGS.config_file}')
logger.info(f'files: {config["files"]}')
logger.info(f'training params: {config["training"]}')


agent = DeepQAgent()
if config['files']['model_file']:
    agent.load_model(config['files']['model_file'])
    head, tail = os.path.split(config['files']['model_file'])
    agent.name = tail
else: agent.name='New model'



################################################


trainer = Trainer(agent=agent, 
                    lr=config['training']['lr'],
                    gamma=config['training']['gamma'],
                    iters=config['training']['iters'],
                    n_episodes=config['training']['n_episodes'], 
                    batch_size=config['training']['batch_size'], 
                    eps=config['training']['eps'])


evaluator = Evaluator()

for epoch in range(config['training']['epochs']):
    logger.info(f'Starting epoch {epoch}')

    trainer.train()

    ## EVALUATE
    ## DECIDE WHAT SHOULD BE TRACKED AND EVALUATED
    ## Should save all models throughout training






#################################################################################
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, default='', help='path to model for training')
#     parser.add_argument('--config_path', type=str, default='config.yaml', help='path to config file')
#     parser.add_argument('--lr', type=float, default=.005, help='learning rate')
#     parser.add_argument('--gamma', type=float, default=.9, help='discount rate')
#     parser.add_argument('--eps', type=float, default=.1, help='epsilon for action selection')
#     parser.add_argument('--epochs', type=int, default=1, help='how many self-play, learn, evaluate cycles')
#     parser.add_argument('--iters', type=int, default=1000, help='how many learning iterations')
#     parser.add_argument('--n_episodes', type=int, default=5, help='how many self-play games per learning iteration')
#     parser.add_argument('--batch_size', type=int, default=64, help='batch size for learning')
#     return parser.parse_args()


