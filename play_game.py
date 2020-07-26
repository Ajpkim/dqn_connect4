import random
from connect4 import Connect4
from deep_q_agent import DeepQAgent
from mdp import Connect4MDP

def play_game(agent_1, agent_2, shuffle_order=True, verbose=False):
    """
    Function for playing a game of Connect4. 
    Necessary to flip the board the q agents playing as player 2.
    Returns winning agents name or 'tie'.
    """
    p1_id = 1
    p2_id = -1

    if shuffle_order: 
        turn = random.choice((p1_id, p2_id))
    else: 
        turn = p1_id
    
    game = Connect4MDP()
    if verbose:
        print('New game!')
    
    while not game.check_game_over():

        if turn == p1_id:
            if verbose:
                print(game)
                print("player 1's turn")
            
            move = agent_1.get_next_move(game.board)
            if move not in game.valid_moves():
                move = random.choice(game.valid_moves())
                print(f'Illegal move. Random move ({move}) chosen instead.')
            game.make_move(move, p1_id)
            turn = p2_id

        elif turn == p2_id:
            if verbose:
                print(game)
                print("player 2's turn")

            if type(agent_2) is DeepQAgent:
                flipped_board = game.get_flipped_board()
                move = agent_2.get_next_move(flipped_board)
            else:
                move = agent_2.get_next_move(game.board)
    
            if move not in game.valid_moves():
                print(f'Illegal move. Random move ({move}) chosen instead.')
                move = random.choice(game.valid_moves())
            game.make_move(move, p2_id)
            turn = p1_id

    if game.status == p1_id:
        outcome = agent_1.name
    elif game.status == p2_id:
        outcome = agent_2.name
    else:
        outcome = 'tie'

    if verbose:
        print(game)
        print('Game Over!')
        if outcome == 'tie':
            print('Tie game!')
        else:
            print(f'Winner: {outcome}')
    
    return outcome
