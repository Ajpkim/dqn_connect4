import logging
import random
import numpy as np


from connect4 import Connect4
from deep_q_agent import DeepQAgent
from logger import setup_logger, get_file_handler, get_logger
from mdp import Connect4MDP
# class for evaluating new model is better than best model and replacing best if so

log_file = 'EVAL.log'
# setup_logger(log_file)
logger = logging.getLogger()
logger = get_logger(__name__, log_file=log_file, level=10)
# logger.setLevel(10)
# logger.addHandler(get_file_handler(log_file))
logger.propagate = False

class Evaluator:
	def __init__(self):
		pass

	def evaluate(self, agent_1, agent_2, n_episodes):
		"Method for comparing 2 agents. Returns dict of totals and dict of percentages."
		
		results, percentages = {}, {}
		outcomes = np.array([self.play_game(agent_1, agent_2) for n in range(n_episodes)])
	
		results[agent_1.name] = (outcomes == agent_1.name).sum()
		results[agent_2.name] = (outcomes == agent_2.name).sum()
		results['tie'] = len(outcomes) - results[agent_1.name] - results[agent_2.name]

		percentages[agent_1.name] = round(results[agent_1.name] / n_episodes, 4)
		percentages[agent_2.name] = round(results[agent_2.name] / n_episodes, 4)
		percentages['tie'] = round(results['tie'] / n_episodes, 4)
		
		return results, percentages




	def play_game(self, agent_1, agent_2):
		"""
		Function to simulate a connect4 game between agent1 and agent2. 
		Need to provide a flipped board to Q Agents which play as player2.

		Returns: name of winning agent, or 'tie'
		"""
		agent_1_id = 1
		agent_2_id = -1
		logger.info(f'Game between {agent_1.name} and {agent_2.name}')

		turn = random.choice((agent_1_id, agent_2_id))
		game = Connect4MDP()
		
		while not game.check_game_over():
			if turn == agent_1_id:
				move = agent_1.get_next_move(game.board)
				
				logger.info(f'\n\n P1 state \n {game.board}')
				logger.info(f'{agent_1.name} move {move}')

				game.make_move(move, agent_1_id)
				turn = agent_2_id

			elif turn == agent_2_id:
				if type(agent_2) is DeepQAgent:
					flipped_board = game.get_flipped_board()
					move = agent_2.get_next_move(flipped_board)	

					logger.info(f'\n\n P2 state \n {flipped_board}')
					logger.info(f'{agent_2.name} move {move}')

				else:
					move = agent_2.get_next_move(game.board)
				game.make_move(move, agent_2_id)
				turn = agent_1_id
		
		if game.status == agent_1_id: 
			return agent_1.name
		elif game.status == agent_2_id: 
			return agent_2.name
		else: return 'tie'


		
		
		


		

