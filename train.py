# brief loop for tesing current setup...
def train(Q_agent, iters, n_episodes):
    mdp = Connect4MDP()
    interval = iters // 10
    scores_v_random = {}

    for i in range(iters):

        if i % interval == 0:
            print('iter:', i)
            scores = eval_players(Q_agent, RandomPlayer(), num_games=1000)
            scores_v_random[i] = scores
            print(
                f'performance vs random agent. wins: {scores["p1"]}, losses: {scores["p2"]}, ties: {scores["tie"]}\n')

        Q_agent.self_play(mdp, n_episodes)
        Q_agent.learn()

    return Q_agent, scores
