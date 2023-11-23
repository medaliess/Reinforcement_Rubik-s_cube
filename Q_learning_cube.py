from Rubiks_Cube_env import RubiksEnv
import numpy as np
import random
from tqdm import tqdm
import pickle

class Q_learning_cube():
    def __init__(self, env, alpha = 0.1, gamma = 0.6, epsilon = 0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((np.prod(env.observation_space.shape)*6, env.action_space.n))
       
        
    def update_q_table(self, old_state, action, reward, new_state):
        self.q_table[old_state, action] = self.q_table[old_state, action] + self.alpha * (reward + self.gamma * np.max(self.q_table[new_state, :]) - self.q_table[old_state, action])
        
    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state, :])
        
    def train(self, num_episodes = 1000):
        rewards_all_episodes = []
        for episode in tqdm(range(num_episodes)):
            shuffles=(episode//1000)+1
            state = self.env.reset(N_shuffle = 1)
            done = False
            rewards_current_episode = 0
            while not done :
                action = self.get_action(state)
                new_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                rewards_current_episode += reward
            print("\n",rewards_current_episode)
            rewards_all_episodes.append(rewards_current_episode)
        
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)
        return rewards_all_episodes
        

    
    def test(self, num_episodes = 100):
        # Load q_table
        with open('q_table.pkl', 'rb') as f:
            self.q_table = pickle.load(f)
        rewards_all_episodes = []
        for episode in range(num_episodes):
            state = self.env.reset(N_shuffle=1)
            done = False
            rewards_current_episode = 0
            while not done:
                action = np.argmax(self.q_table[state, :])
                new_state, reward, done = self.env.step(action)
                state = new_state
                rewards_current_episode += reward
            rewards_all_episodes.append(rewards_current_episode)
           
        # rander
        self.env.render()
        return rewards_all_episodes
    

env = RubiksEnv()
agent = Q_learning_cube(env)
reward=agent.train()
print(reward)

