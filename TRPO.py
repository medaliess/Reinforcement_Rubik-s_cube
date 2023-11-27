from Cube_env import RubiksEnv
import numpy as np
import random
from tqdm import tqdm
import pickle
import tensorflow as tf
from tensorflow import keras

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = keras.layers.Dense(64, activation='relu')
        self.fc3 = keras.layers.Dense(action_dim)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc3(x)
        return x

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        super(Actor, self).__init__()
        self.fc1 = keras.layers.Dense(64, activation='relu')
        self.fc3 = keras.layers.Dense(action_dim, activation='softmax')
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc3(x)
        return x


class TRPO():

    def __init__(self, env, gamma=0.9, max_kl=0.01, damping=0.1):
        self.env = env
        self.gamma = gamma
        self.max_kl = max_kl
        self.damping = damping
        self.state_dim = len(self.env.observation_space.sample())
        self.action_dim = self.env.action_space.n
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.alpha_critic=0.001

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor(state)
        action = np.random.choice(self.action_dim, p=np.squeeze(action_probs))
        return action

    def get_Q_values(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        return self.critic(state).numpy()[0]

    def calculate_advantage(self, state, delta):
        state= tf.convert_to_tensor([state], dtype=tf.float32)
        y = np.sum(self.actor(state).numpy()[0] * self.get_Q_values(state))
        return delta.numpy()[0] - y

    def calculate_delta(self, state, action, reward, next_state):
        state= tf.convert_to_tensor([state], dtype=tf.float32)
        next_state= tf.convert_to_tensor([next_state], dtype=tf.float32)
        reward= tf.convert_to_tensor([reward], dtype=tf.float32)
        q_value = self.critic(state)[0][action]
        next_q_value = self.critic(next_state)[0]
        delta = reward + self.gamma * tf.reduce_max(next_q_value) - q_value

        return delta
    
    def update_critic(self, state, action, delta, n):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        with tf.GradientTape() as tape:
            q_value = self.critic(state)[0][action]
            target_value = delta + q_value
            loss = tf.losses.mean_squared_error(target_value, self.critic(state)[0][action])

        gradients = tape.gradient(loss, self.critic.trainable_variables)

        for i in range(len(gradients)):
            self.critic.trainable_variables[i].assign_add(self.alpha_critic * self.gamma ** n * gradients[i])

    def update_actor(self, states, actions, deltas, advantages):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = np.array(actions)
        advantages = np.array(advantages, dtype=np.float32)

        with tf.GradientTape() as tape:
            probs = self.actor(states)

            ratio = tf.reduce_sum(probs * tf.one_hot(actions, depth=self.action_dim), axis=1) / (
                    tf.reduce_sum(probs * tf.one_hot(actions, depth=self.action_dim), axis=1) + 1e-8)

            surrogate = ratio * advantages

            # Experiment with the entropy regularization term
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
            loss = -tf.reduce_mean(surrogate - 0.01 * entropy)

        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
        

env = RubiksEnv() 
agent = TRPO(env)

num_episodes = 3000  

for episode in tqdm(range(num_episodes)):
    shuffles=episode//1000 +1
    state = env.reset(N_shuffle=shuffles)

    states, actions, rewards,deltas,advantages = [], [], [], [], []

    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        deltas.append(agent.calculate_delta(state, action, reward, next_state))
        advantages.append(agent.calculate_advantage(state, deltas[-1]))
        #update critic
        agent.update_critic(state, action, deltas[-1], len(states))
        state = next_state
    print("Reward: {}".format(np.sum(rewards)))
    # update actor
    if states:
        agent.update_actor(states, actions,deltas[-1], advantages)

    