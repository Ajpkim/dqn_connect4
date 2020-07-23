import random

from connect4 import Connect4
# class for evaluating new model is better than best model and replacing best if so

class Evaluator:
	def __init__(self):
		pass

	def evaluate(self, agent_1, agent_2, n_episodes):
		"Method for comparing 2 agents. Returns dict of totals and dict of percentages."
		percentages = {}
		results = {agent_1.name: 0, agent_2.name: 0, 'tie': 0}
		for n in range(n_episodes):
			outcome = self.play_game(agent_1, agent_2)
			if outcome == 1:
				results[agent_1.name] += 1
			elif outcome == -1:
				results[agent_2.name] += 1
			elif outcome == 'tie': 
				results['tie'] += 1
		
		percentages[agent_1.name] = round(results[agent_1.name] / n_episodes, 4)
		percentages[agent_2.name] = round(results[agent_2.name] / n_episodes, 4)
		percentages['tie'] = round(results['tie'] / n_episodes, 4)
		
		return results, percentages

	def play_game(self, agent_1, agent_2):
		"Return 1 for p1 win, -1 for p2 win, 0 for tie"
		
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

		
		
		


		

