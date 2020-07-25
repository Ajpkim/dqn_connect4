import numpy as np

from logger import *
from mdp import Connect4MDP
from deep_q_agent import DeepQAgent

def self_play_episodes(mdp: Connect4MDP, agent: DeepQAgent, episodes: int, eps: float) -> list:
    """
    Generate training date through self play. 
    Return list of experience tuples: (state, action, reward, next_state)
    """
    p1_id = 1
    p2_id = -1
    all_experiences = []

    for i in range(episodes):
        mdp.reset()
        states_actions_rewards = []
        state = mdp.get_state()   
        turn = p1_id

        while not mdp.check_game_over():
            if turn == p1_id:
                action = agent.select_action(state, eps, mdp.valid_moves())
                states_actions_rewards.append((state, action, 0))
                state = mdp.make_move(action, id=p1_id)
                turn = p2_id
        
            elif turn == p2_id:
                state = mdp.get_flipped_state()
                action = agent.select_action(state, eps, mdp.valid_moves())
                states_actions_rewards.append((state, action, 0))
                state = mdp.make_move(action, id=p2_id)
                turn = p1_id

    # GAME OVER. Gather experience tuples: (state, action, reward, next_state, done)
        # Update final entries with rewards wrt game outcome
        p1_reward = mdp.reward_fn(id=p1_id)
        p2_reward = mdp.reward_fn(id=p2_id)
        ### PASSING REWARD TO ALL EXPERIENCES IN GAME ... p1 rewards to even turns, p2 to odd
        states_actions_rewards[::2] = map(lambda tup: tup[0:2] + (p1_reward,), states_actions_rewards[::2])
        states_actions_rewards[1::2] = map(lambda tup: tup[0:2] + (p2_reward,), states_actions_rewards[1::2])

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

########### Passing reward only to terminal experiences
        
        # if len(states_actions_rewards) % 2 != 0:  # add reward to correct sequence
        #     states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p1_reward,)
        #     states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p2_reward,)
        # else:
        #     states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p2_reward,)
        #     states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p1_reward,)










# def self_play_episodes(mdp: Connect4MDP, agent: DeepQAgent, episodes: int, eps: float) -> list:
#     """
#     Generate training date through self play. 
#     Return list of experience tuples: (state, action, reward, next_state)
#     """
#     p1_id = 1
#     p2_id = -1
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
        
#         ### PASSING REWARD TO ALL EXPERIENCES IN GAME ... p1 rewards to even turns, p2 to odd
        
#         states_actions_rewards[::2] = map(lambda states_actions: states_actions + (p1_reward,), states_actions_rewards[::2])
#         states_actions_rewards[1::2] = map(lambda states_actions: states_actions + (p2_reward,), states_actions_rewards[1::2])

#         # if len(states_actions_rewards) % 2 != 0:  # add reward to correct sequence
#         #     states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p1_reward,)
#         #     states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p2_reward,)
#         # else:
#         #     states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p2_reward,)
#         #     states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p1_reward,)


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
########### Passing reward only to terminal experiences
        
        # if len(states_actions_rewards) % 2 != 0:  # add reward to correct sequence
        #     states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p1_reward,)
        #     states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p2_reward,)
        # else:
        #     states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p2_reward,)
        #     states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p1_reward,)



################################################################################
### DEBUGGIN CALLS BELOW ###
# setup_logger('SELF_PLAY_TESTING.log')
# logger = logging.getLogger(__name__)

# logger.info('Self play logging')



# def self_play_episodes(mdp: Connect4MDP, agent: DeepQAgent, episodes: int, eps: float) -> list:
#     """
#     Generate training date through self play. 
#     Return list of experience tuples: (state, action, reward, next_state)
#     """
#     p1_id = 1
#     p2_id = -1
#     all_experiences = []

#     for i in range(episodes):
#         mdp.reset()
#         states_actions = []
#         state = mdp.get_state()   
        
        
        
#         turn = p1_id

#         while not mdp.check_game_over():
#             if turn == p1_id:
                
                
                
#                 action = agent.select_action(state, eps, mdp.valid_moves())
                
                
#                 logger.info(f'\n\nP1 state \n{state.reshape(6,7)}')
#                 logger.info(f'P1 action {action}')


#                 states_actions.append((state, action))
#                 state = mdp.make_move(action, id=p1_id)
#                 turn = p2_id
        
#             elif turn == p2_id:

#                 state = flip_state(state)
#                 action = agent.select_action(state, eps, mdp.valid_moves())
                
                
#                 logger.info(f'\n\nP2 state \n{state.reshape(6,7)}')
#                 logger.info(f'P2 action {action}')


#                 states_actions.append((state, action))
#                 state = mdp.make_move(action, id=p2_id)
#                 turn = p1_id


#     # GAME OVER. Gather experience tuples: (state, action, reward, next_state, done)
#         # Update final entries with rewards wrt game outcome
#         p1_reward = mdp.reward_fn(id=p1_id)
#         p2_reward = mdp.reward_fn(id=p2_id)
#         ### PASSING REWARD TO ALL EXPERIENCES IN GAME ... p1 rewards to even turns, p2 to odd
#         states_actions[::2] = map(lambda tup: tup[0:2] + (p1_reward,), states_actions[::2])
#         states_actions[1::2] = map(lambda tup: tup[0:2] + (p2_reward,), states_actions[1::2])

#         # new_state is from player POV i.e. p1 sees turn 0, turn 2, turn 4 board states, etc.
#         states = [x[0] for x in states_actions]
#         states += [states[0], states[0]]  # add blank states as next_states following terminal states... will be ignored when learning

#         # Done flag for terminal states
#         dones = [False for i in range(len(states_actions)-2)]
#         dones += [True, True]  
        
#         states_actions_rewards_new_states_dones = [(states_actions[i] + (states[i+2],dones[i])) 
#                                                     for i in range(len(states_actions))]
#         all_experiences += states_actions_rewards_new_states_dones
    
#         for s,a,r,ns,d in all_experiences[:5]:
#             print('-------------------')
#             print(s.reshape(6,7))
#             print(a)
#             print(r)
#             print(ns.reshape(6,7))
#             print()
            



#     return all_experiences

        ### Passing reward only to terminal experiences
        # if len(states_actions_rewards) % 2 != 0:  # add reward to correct sequence
        #     states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p1_reward,)
        #     states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p2_reward,)
        # else:
        #     states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p2_reward,)
        #     states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p1_reward,)








#### PRE FLIP BOARD #########

# def self_play_episodes(mdp: Connect4MDP, agent: DeepQAgent, episodes: int, eps: float) -> list:
#     """
#     Generate training date through self play. 
#     Return list of experience tuples: (state, action, reward, next_state)
#     """

#     p1_id = 1
#     p2_id = -1

#     all_experiences = []
#     for i in range(episodes):

#         mdp.reset()
#         states_actions = []
#         state = mdp.get_state()
        
#         # while mdp.status == 0:
#         while not mdp.check_game_over():
            
#             id = p1_id
#             action = agent.select_action(mdp, eps)
#             states_actions.append((state, action))
#             state = mdp.make_move(action, id=id)

#             if mdp.status != 0: break
            
#             id = p2_id                     
#             action = agent.select_action(mdp, eps)
#             states_actions.append((state, action, 0))  # 0 reward for all states except last
#             state = mdp.make_move(action, id=id)

#     # GAME OVER. Gather experience tuples: (state, action, reward, next_state, done)
#         # Update final entries with rewards wrt game outcome
#         p1_reward = mdp.reward_fn(id=p1_id)
#         p2_reward = mdp.reward_fn(id=p2_id)
#         ### PASSING REWARD TO ALL EXPERIENCES IN GAME ... p1 rewards to even turns, p2 to odd
#         states_actions[::2] = map(lambda tup: tup[0:2] + (p1_reward,), states_actions[::2])
#         states_actions[1::2] = map(lambda tup: tup[0:2] + (p2_reward,), states_actions[1::2])

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

#         ## Passing reward only to terminal experiences
#         if len(states_actions_rewards) % 2 != 0:  # add reward to correct sequence
#             states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p1_reward,)
#             states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p2_reward,)
#         else:
#             states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p2_reward,)
#             states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p1_reward,)
