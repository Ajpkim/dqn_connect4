import numpy as np

import torch
import torch.nn as nn

from deep_q_agent import DeepQAgent
from self_play_episodes import self_play_episodes



class Trainer:
    """
    ... 
    """
    def __init__(self, mdp, agent, lr, gamma, iters, n_episodes, batch_size, eps):
        self.mdp = mdp
        self.agent = agent
        self.optimizer = torch.optim.Adam(params=agent.net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.gamma = gamma
        self.iters = iters
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.eps = eps
    
    def self_play(self, episodes):
    # def self_play(self, mdp=self.mdp, agent=self.agent, n_episodes=self.n_episodes):
        """
        Generate training data by playing games vs self.
        Gathers experiece tuples over n_episodes and pushes them to agent replay buffer.
        """
        experiences = self_play_episodes(self.mdp, self.agent, episodes, self.eps)
        for state, action, reward, next_state, done in experiences:
            self.agent.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """
        Update agent network with random batch from agent replay buffer.
        """
        batch = self.agent.replay_buffer.sample(self.batch_size)
        states = torch.tensor([x.state for x in batch]).float().to(self.agent.device)
        actions = [x.action for x in batch]
        rewards = torch.tensor([x.reward for x in batch]).float().to(self.agent.device)
        next_states = torch.tensor([x.next_state for x in batch]).float().to(self.agent.device)
        dones = [x.done for x in batch]

        self.optimizer.zero_grad()
        q_vals = self.agent.net(states)[range(len(actions)), actions]
        q_next_vals = self.agent.net(next_states)
        q_next_vals[dones] = 0
        q_targets = rewards + self.gamma * torch.max(q_next_vals, dim=1)[0]

        loss = self.loss_fn(q_targets, q_vals).to(self.agent.device)
        loss.backward()
        self.optimizer.step()

    def train(self):  ## ADD IN ITERS ARG?
        """
        Train agent over given number of iterations.
        Each iteration consists of self play over self.n_episodes 
        and then a learn step where agent updates network based on 
        random sample from replay buffer
        """
        for i in range(self.iters):
            self.self_play(self.n_episodes)
            self.learn()



########



    # def self_play(mdp=self.mdp, agent=self.agent, n_episodes=self.n_episodes) -> list:
    # """
    # Generate training date through self play. 
    # Return list of experience tuples: (state, action, reward, next_state)
    # """
    # p1_id = 1
    # p2_id = -1
    # all_experiences = []

    # for i in range(episodes):
    #     mdp.reset()
    #     states_actions_rewards = []
    #     state = mdp.get_state()
            
    #     while mdp.status == 0:
    #         id = p1_id
    #         action = agent.select_action(mdp)
    #         states_actions_rewards.append((state, action, 0))  # 0 reward for all states except last
    #         state = mdp.make_move(action, id=id)

    #         if mdp.status != 0: break
            
    #         id = p2_id                     
    #         action = agent.select_action(mdp)
    #         states_actions_rewards.append((state, action, 0))  # 0 reward for all states except last
    #         state = mdp.make_move(action, id=id)

    # # GAME OVER. Gather experience tuples: (state, action, reward, next_state, done)
    #     # Update final entries with rewards wrt game outcome
    #     p1_reward = mdp.reward_fn(id=p1_id)
    #     p2_reward = mdp.reward_fn(id=p2_id)
        
    #     if len(states_actions_rewards) % 2 != 0:  # add reward to correct sequence
    #         states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p1_reward,)
    #         states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p2_reward,)
    #     else:
    #         states_actions_rewards[-1] = states_actions_rewards[-1][0:2] + (p2_reward,)
    #         states_actions_rewards[-2] = states_actions_rewards[-2][0:2] + (p1_reward,)

    #     # new_state is from player POV i.e. p1 sees turn 0, turn 2, turn 4 board states, etc.
    #     states = [x[0] for x in states_actions_rewards]
    #     states += [states[0], states[0]]  # add blank states as next_states following terminal states

    #     # Done flag for terminal states
    #     dones = [False for i in range(len(states_actions_rewards)-2)]
    #     dones += [True, True]  
        
    #     states_actions_rewards_new_states_dones = [(states_actions_rewards[i] + (states[i+2],dones[i])) 
    #                                                 for i in range(len(states_actions_rewards))]
    #     all_experiences += states_actions_rewards_new_states_dones

    
    # return all_experiences