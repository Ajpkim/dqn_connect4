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
		
		return results

	def play_game(self, agent_1, agent_2):
		"Return 1 for p1 win, 2 for p2 win, 0 for tie"
		
		p1_id = 1
		p2_id = -1

		turn = random.choice((p1_id, p2_id))
		game = Connect4()
		
		while game.status == 0:

			if turn == p1_id:
				move = agent_1.get_next_move(game.board)
				game.make_move(move, p1_id)
				turn = p2_id

			elif turn == p2_id:
				move = agent_2.get_next_move(game.board)
				game.make_move(move, p2_id)
				turn = p1_id

		return game.status

		
		
		


		

