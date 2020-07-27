import numpy as np

from logger import *
from mdp import Connect4MDP
from deep_q_agent import DeepQAgent



### SWAPPING TO NO BOARD FLIPPING AND TO HANDLING STATE WITHIN AGENT CLASS
def self_play_episodes(mdp: Connect4MDP, agent: DeepQAgent, episodes: int, eps: float) -> list:
    """
    Generate training date through self play. 
    Return list of experience tuples: (state, action, reward, next_state)
    """
    p1_id = 1
    p2_id = 2
    all_experiences = []

    for i in range(episodes):
        mdp.reset()
        states_actions_rewards = []
        turn = p1_id
       
        while not mdp.check_game_over():
            if turn == p1_id:
                state = agent.encode_board(mdp.board)
                action = agent.select_action(state, eps, mdp.valid_moves())
                states_actions_rewards.append((state, action, 0))
                state = mdp.make_move(action, id=p1_id)
                turn = p2_id
        
            elif turn == p2_id:
                state = agent.encode_board(mdp.board)
                action = agent.select_action(state, eps, mdp.valid_moves())
                states_actions_rewards.append((state, action, 0))
                state = mdp.make_move(action, id=p2_id)
                turn = p1_id

    # GAME OVER. Gather experience tuples: (state, action, reward, next_state, done)
        p1_reward = mdp.reward_fn(id=p1_id)
        p2_reward = mdp.reward_fn(id=p2_id)

###################### Passing reward only to terminal experiences        
        if len(states_actions_rewards) % 2 != 0:  # add reward to correct sequence
            states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p1_reward,)
            states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p2_reward,)
        else:
            states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p2_reward,)
            states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p1_reward,)
############### PASSING REWARD TO ALL EXPERIENCES IN GAME ... p1 rewards to even turns, p2 to odd
        # states_actions_rewards[::2] = map(lambda tup: tup[0:2] + (p1_reward,), states_actions_rewards[::2])
        # states_actions_rewards[1::2] = map(lambda tup: tup[0:2] + (p2_reward,), states_actions_rewards[1::2])
######################

        # new_state is from player POV i.e. p1 sees turn 0, turn 2, turn 4 board states, etc.
        states = [x[0] for x in states_actions_rewards]
        states += [states[0], states[0]]  # add blank states as next_states following terminal states... will be ignored when learning

        # Done flag for terminal states
        dones = [False for i in range(len(states_actions_rewards)-2)]
        dones += [True, True]  
        
        states_actions_rewards_new_states_dones = [(states_actions_rewards[i] + (states[i+2],dones[i])) 
                                                    for i in range(len(states_actions_rewards))]
        all_experiences += states_actions_rewards_new_states_dones
    
    return all_experiences



####################################################
### PRE CNN
# def self_play_episodes(mdp: Connect4MDP, agent: DeepQAgent, episodes: int, eps: float) -> list:
#     """
#     Generate training date through self play. 
#     Return list of experience tuples: (state, action, reward, next_state)
#     """
#     p1_id = 1
#     p2_id = 2
#     all_experiences = []

#     for i in range(episodes):
#         mdp.reset()
#         states_actions_rewards = []
#         state = mdp.get_state()   
#         turn = p1_id
       
#         while not mdp.check_game_over():
#             if turn == p1_id:
#                 action = agent.select_action(state, eps, mdp.valid_moves())
#                 states_actions_rewards.append((state, action, 0))
#                 state = mdp.make_move(action, id=p1_id)
#                 turn = p2_id
        
#             elif turn == p2_id:
#                 state = mdp.get_flipped_state()
#                 action = agent.select_action(state, eps, mdp.valid_moves())
#                 states_actions_rewards.append((state, action, 0))
#                 state = mdp.make_move(action, id=p2_id)
#                 turn = p1_id

#     # GAME OVER. Gather experience tuples: (state, action, reward, next_state, done)
#         # Update final entries with rewards wrt game outcome
#         p1_reward = mdp.reward_fn(id=p1_id)
#         p2_reward = mdp.reward_fn(id=p2_id)

# ###################### Passing reward only to terminal experiences        
#         if len(states_actions_rewards) % 2 != 0:  # add reward to correct sequence
#             states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p1_reward,)
#             states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p2_reward,)
#         else:
#             states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p2_reward,)
#             states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p1_reward,)
# ############### PASSING REWARD TO ALL EXPERIENCES IN GAME ... p1 rewards to even turns, p2 to odd
#         # states_actions_rewards[::2] = map(lambda tup: tup[0:2] + (p1_reward,), states_actions_rewards[::2])
#         # states_actions_rewards[1::2] = map(lambda tup: tup[0:2] + (p2_reward,), states_actions_rewards[1::2])
# ######################

#         # new_state is from player POV i.e. p1 sees turn 0, turn 2, turn 4 board states, etc.
#         states = [x[0] for x in states_actions_rewards]
#         states += [states[0], states[0]]  # add blank states as next_states following terminal states... will be ignored when learning

#         # Done flag for terminal states
#         dones = [False for i in range(len(states_actions_rewards)-2)]
#         dones += [True, True]  
        
#         states_actions_rewards_new_states_dones = [(states_actions_rewards[i] + (states[i+2],dones[i])) 
#                                                     for i in range(len(states_actions_rewards))]
#         all_experiences += states_actions_rewards_new_states_dones
    
#     return all_experiences






