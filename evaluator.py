import random

from connect4 import Connect4
# class for evaluating new model is better than best model and replacing best if so

class Evaluator:
	def __init__(self):
		pass

	def evaluate(self, best_agent, new_agent, n_episodes):
		"Method for comparing best agent vs new agent"
		
		results = {'best': 0, 'new': 0, 'tie': 0}
		for n in range(n_episodes):
			outcome = self.play_game(best_agent, new_agent)
			if outcome == 1:
				results['best'] += 1
			elif outcome == -1:
				results['new'] += 1
			else: results['tie'] += 1
		
		# if results['new'] > results['best']:
		# 	return True
		# else: return False
		
		return results

	def play_game(self, agent_1, agent_2):
		"Return 1 for p1 win, 2 for p2 win, 0 for tie"
		
		agent_1_id = 1
		agent_2_id = -1

		turn = random.choice((agent_1_id, agent_2_id))
		game = Connect4()
		
		while game.status == 0:

			if turn == agent_1_id:
				move = agent_1.get_next_move(game.board)
				game.make_move(move, agent_1_id)
				turn = agent_2_id

			elif turn == agent_2_id:
				move = agent_2.get_next_move(game.board)
				game.make_move(move, agent_2_id)
				turn = agent_1_id

		return game.status

		
		
		


		

