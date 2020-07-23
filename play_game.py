import random
from connect4 import Connect4

def play_game(p1, p2, shuffle_order=True, verbose=False):
    "Return 1 for p1 win, 2 for p2 win, 0 for tie"
    p1_id = 1
    p2_id = -1

    if shuffle_order: 
        turn = random.choice((p1_id, p2_id))
    else: 
        turn = p1_id
    
    game = Connect4()
    if verbose:
        print('New game!')
    
    while game.status == 0:

        if turn == p1_id:
            if verbose:
                print(game)
                print("player 1's turn")
            
            move = p1.get_next_move(game.board)
            if move not in game.valid_moves():
                move = random.choice(game.valid_moves())
                print(f'Illegal move. Random move ({move}) chosen instead.')
            game.make_move(move, p1_id)
            turn = p2_id

        elif turn == p2_id:
            if verbose:
                print(game)
                print("player 2's turn")

            move = p2.get_next_move(game.board)
            if move not in game.valid_moves():
                print(f'Illegal move. Random move ({move}) chosen instead.')
                move = random.choice(game.valid_moves())
            game.make_move(move, p2_id)
            turn = p1_id

    if verbose:
        print(game)
        print('Game Over!')
    
    return game.status
