from Rubiks_Cube_env import RubiksEnv
import numpy as np
import random
from tqdm import tqdm
import pickle

class Q_learning_cube():
    def __init__(self, env, alpha = 0.01, gamma = 0.9, epsilon = 1, epsilon_decay = 0.0001):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
       
    def get_Q_value(self,state):
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.env.action_space.n)
        return self.q_table[state_key]
    def update_q_table(self, state, action, reward, new_state):
        state_key = tuple(state)
        old_value = self.get_Q_value(state)[action]
        next_max = np.max(self.get_Q_value(new_state)) 
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state_key][action] = new_value

    def get_action(self, state):
        state_key = tuple(state)
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
           if state_key not in self.q_table:
               self.q_table[state_key] = np.zeros(self.env.action_space.n)
               return self.env.action_space.sample()
           return np.argmax(self.q_table[state_key]) 

    
    def train(self):
        rewards_all_episodes = []
        i=1
        while i <6 :
            self.epsilon = 1
            for episode in tqdm(range(10000*2*i)):
                state = self.env.reset(N_shuffle=i)
                done = False
                rewards_current_episode = 0
                while not done:
                    action = self.get_action(state)
                    new_state, reward, done = self.env.step(action)
                    self.update_q_table(state, action, reward, new_state)
                    state = new_state
                    rewards_current_episode += reward
                rewards_all_episodes.append(rewards_current_episode)
                self.epsilon -= self.epsilon_decay/(2*i)
            i+=1
            print('epsilone',self.epsilon)
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)
        return rewards_all_episodes
        

    
    def test(self, num_episodes = 100):
        # Load q_table
        with open('q_table.pkl', 'rb') as f:
            self.q_table = pickle.load(f)
        self.epsilon = 0
        i=1
        while i < 6:
            rewards_all_episodes = []
            for episode in tqdm(range(100)):
                state = self.env.reset(N_shuffle=i)
                done = False
                rewards_current_episode = 0
                while not done:
                    action = self.get_action(state)
                    new_state, reward, done = self.env.step(action)
                    state = new_state
                    rewards_current_episode += reward
                rewards_all_episodes.append(rewards_current_episode)
            #count negatife rewards and print percentage of solved cubes shuffled i moves
            print("percentage of solved cubes shuffled ",i,"moves :",100*len([reward for reward in rewards_all_episodes if reward > 0])/len(rewards_all_episodes),"%")
            i+=1
            self.env.render()
            
            
if __name__ == "__main__":       
    env = RubiksEnv()
    agent = Q_learning_cube(env)
    reward=agent.test()


