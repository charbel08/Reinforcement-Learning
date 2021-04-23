#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import gym

# This function measures performance of a random policy in an environment env
def random_actions(env, env_name, episodes, goal=None, plot=True):
    rewards = []
    for episode in range(episodes):
        
        state = env.reset()
        done = False
        episode_rewards = []
        
        while not done:
            # Sample random actions
            action = env.action_space.sample()
            
            # Take action
            state, reward, done, _ = env.step(action)
            
            # Update episode rewards
            episode_rewards.append(reward)
            
        # Add to the mean episode rewards
        rewards.append(np.sum(episode_rewards))
        
    if plot:
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5))
        plt.title("Random policy rewards for " + env_name)
        ax.plot(range(len(rewards)), rewards, label='rewards')
        if goal != None:
            ax.axhline(goal, c='red',ls='--', label='goal')
        ax.legend()
        plt.show()
        

# This function provides the average score and standard deviation over 100 consecutive runs
def final_benchmark(env, agent, runs, render=False):
    # Final Benchmarking
    rewards = np.zeros((runs))
    for run in range(runs):
        obs = env.reset()
        done = False
        episodes_rewards = 0
        while not done:
            if render:
                env.render()
            act = agent.policy(obs)
            obs, rew, done, info = env.step(act)
            episodes_rewards += rew
        rewards[run] = episodes_rewards
        
    print("For 100 different episodes:")
    print("Average return: {}".format(rewards.mean()))
    print("Standard deviation: {}".format(rewards.std()))
    

class DQN(nn.Module):
    def __init__(self, lr, input_dims, action_dim):
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.action_dim = action_dim
        self.fc1 = nn.Linear(*self.input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions
    

class DQNAgent():
    def __init__(self, env, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.05, 
                 decay=5e-4):

        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.lr = lr
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.e = epsilon

        self.q = DQN(lr, action_dim=n_actions, input_dims=input_dims)
        self.target = DQN(lr, action_dim=n_actions, input_dims=input_dims)

        self.replay_buffer = deque(maxlen=10000)

    def epsilon_greedy(self, obs):
        # Follow policy
        if random.uniform(0, 1) > self.epsilon:
            state = torch.Tensor([obs])
            return self.q(state).argmax().item()
        # Choose random action
        else:
            return self.env.action_space.sample()

    def policy(self, obs):
        state = torch.Tensor([obs])
        return self.q(state).argmax().item()

    def learn(self, obs):

        self.mem_cntr += 1
        if self.mem_cntr < self.batch_size:
            return

        # Sample from reply buffer
        self.replay_buffer.append(obs)
        samples = random.choices(self.replay_buffer, k=self.batch_size)
        states, actions, rewards, next_states, dones = (np.stack(i) for i in zip(*samples))        

        self.q.optimizer.zero_grad()

        # Batch indices
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Get prediction
        q_eval = self.q(torch.Tensor(states))[batch_index, actions]

        # Get Q target
        q_next = self.target(torch.Tensor(next_states))
        q_next[dones] = 0.0
        q_target = torch.tensor(rewards) + self.gamma*torch.max(q_next,dim=1)[0]

        # Backpropagation
        loss = self.q.loss(q_target, q_eval)
        loss.backward()
        self.q.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon*self.decay, 0.01)

    def train(self, goal=None, C=10, num_episodes=150, render=False, plot=False, verbose=False):

        rewards = []
        for t in range(num_episodes):

            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Follow epsilon greedy policy
                action = self.epsilon_greedy(state)

                # Take action
                next_state, reward, done, _ = self.env.step(action)

                # Update episode rewards
                episode_reward += reward

                # Backpropagate the neural network
                self.learn((state, action, reward, next_state, done))

                # Update state
                state = next_state

            if verbose:
                print("Episode", t, "reward:", episode_reward)

            # Update target network parameters
            if t % C == 0:
                with torch.no_grad():
                    self.target.load_state_dict(self.q.state_dict())

            # Save reward info for this episode
            rewards.append(episode_reward)

        if plot:

            f, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5))
            plt.title("Sum of rewards per episode \n lr=" + str(self.lr) + ", episodes=" + str(num_episodes)
                     + ", batch_size=" + str(self.batch_size) + ", gamma=" + str(self.gamma) + "\n"
                     + "C=" + str(C) + ", epsilon=" + str(self.e))
            ax.plot(range(len(rewards)), rewards, label='rewards')
            if goal != None:
                ax.axhline(goal, c='red',ls='--', label='goal')
            ax.legend()
            plt.show()

        return rewards
    

env = gym.envs.make("MountainCar-v0")
dqn_agent = DQNAgent(env, gamma=0.99, epsilon=1.0, batch_size=64, 
                 n_actions=3, decay=0.999, input_dims=[2], lr=0.003)

dqn_agent.train(195, num_episodes=1200, C=1, verbose=True, plot=False)
final_benchmark(env, dqn_agent, 10, render=True)

env.close()










    
