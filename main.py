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

### Decide if i want file paths in config file or as command line args

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default='config/default.yaml', help='Configuration file location.')
parser.add_argument("--load_file", default='', help='file to load model from.')
parser.add_argument("--save_file", default='models/new_model_in_training', help='file to save model to.')
parser.add_argument("--log_file", default='logs/new_model_logs.log', help='log file location. Corresponds with model being trained.')
ARGS = parser.parse_args()

with open(ARGS.config_file, mode='r') as f:
    config = yaml.safe_load(f)

# HOW CAN I DO THIS W/O CALLING BASICCONFIG()?
logging.basicConfig(filename=ARGS.log_file, 
                    filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

logger.info('\n\n----------   NEW LOG   ----------\n')
logger.info(f'config_file: {ARGS.config_file}')
logger.info(f'load_file: {ARGS.load_file}')
logger.info(f'save_file: {ARGS.save_file}')
logger.info(f'training params: {config}')


agent = DeepQAgent()
if ARGS.load_file:
    agent.load_model(ARGS.load_file)
    head, tail = os.path.split(ARGS.save_file)
    agent.name = tail
else: agent.name='New model'

################################################
trainer = Trainer(agent=agent, 
                    lr=config['lr'],
                    gamma=config['gamma'],
                    iters=config['iters'],
                    n_episodes=config['n_episodes'], 
                    batch_size=config['batch_size'], 
                    eps=config['eps'])

evaluator = Evaluator()

for epoch in range(config['epochs']):
    logger.info(f'Starting epoch {epoch}')

    trainer.train()

    ## EVALUATE
    ## DECIDE WHAT SHOULD BE TRACKED AND EVALUATED
    ## Should save all models throughout training




