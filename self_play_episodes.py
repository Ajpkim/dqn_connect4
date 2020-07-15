def self_play_episodes(mdp: Connect4MDP, agent: DeepQAgent, episodes: int) -> list:
    """
    Generate training date through self play. 
    Return list of experience tuples: (state, action, reward, next_state)
    """

    all_experiences = []
    for i in range(episodes):

        mdp.reset()
        states_actions_rewards = []
        state = mdp.board2vec()
            
        while mdp.status == 0:
            # player 1 turn
            id = 1

            # epsilon greedy action selection
            if random.random() < agent.eps:
                action = random.choice(mdp.valid_moves())
            else:
                invalid_moves = mdp.invalid_moves()
                action_estimates = agent.action_estimates(state)        
                action_estimates[invalid_moves] = -float('inf')  # use invalid_moves as indices to filter out illegal moves
                action = torch.argmax(action_estimates).item() 
            
            states_actions_rewards.append((state, action, 0))  # 0 reward for all states except last
            state = mdp.make_move(action, id=id)

            if mdp.status != 0: break
            
            # player 2 turn
            id = -1                        
            if random.random() < agent.eps:
                action = random.choice(mdp.valid_moves())
            else:
                invalid_moves = mdp.invalid_moves()
                action_estimates = agent.action_estimates(state)        
                action_estimates[invalid_moves] = -float('inf')  
                action = torch.argmax(action_estimates).item() 


            
            states_actions_rewards.append((state, action, 0))  # 0 reward for all states except last
            state = mdp.make_move(action, id=id)


    # GAME OVER. Gather experience tuples: (state, action, reward, next_state, done)
        # Update final entries with rewards wrt game outcome
        p1_reward = mdp.reward_fn(id=1)
        p2_reward = mdp.reward_fn(id=2)

        if len(states_actions_rewards) % 2 != 0:  # add reward to correct sequence
            states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p1_reward,)
            states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p2_reward,)
        else:
            states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p2_reward,)
            states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p1_reward,)

        # new_state is from player POV i.e. p1 sees turn 0, turn 2, turn 4 board states, etc.
        states = [x[0] for x in states_actions_rewards]
        states += [states[0], states[0]]  # add blank states as next_states following terminal states

        # Done flag for terminal states
        dones = [False for state in states_actions_rewards[:-2]]
        dones += [True, True]  
        
        states_actions_rewards_new_states_dones = [(states_actions_rewards[i] + (states[i+2],dones[i])) 
                                                    for i in range(len(states_actions_rewards))]
        all_experiences += states_actions_rewards_new_states_dones
    
    return all_experiences
