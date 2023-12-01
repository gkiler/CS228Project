import gymnasium as gym
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten
from keras.optimizers import Adam
from collections import deque


np.set_printoptions(threshold=sys.maxsize)


class DQNAgent:
    def __init__(self, actions, states):
        self.actions = actions
        self.states = states
        self.memory = deque(maxlen=50000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.gamma = 0.95
        self.model = self.build_model()
        

    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(self.states)))
        model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
        model.add(Convolution2D(64, (3,3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.actions, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
        return model
    
    def buffer(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
# Exploration-exploitation strategy to choose an action based on epsilon-greedy    
    def learn_action(self, state):
        if random.uniform(0,1) <= self.epsilon:
            return np.random.choice(self.actions)
        
        else:
            value = self.model.predict(state)
            return np.argmax(value[0]) # highest Q-value
        
# Learning from a minibatch of using the Q-learning
    def learn_model(self, batch_size=32):
        minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))
        
        states = []
        target_fs = []

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done: 
                target += self.gamma * np.amax(self.model.predict(next_state.reshape(1, *self.states)))
  
            target_f = self.model.predict(state.reshape(1, *self.states))
            target_f[0][action] = target
            states.append(state)
            target_fs.append(target_f)

        states = np.array(states)
        target_fs = np.array(target_fs)

        history = self.model.fit(x=states, y=target_fs, epochs=1, verbose=0)

        loss = history.history['loss'][0]
        #print(f"Loss: {loss}")

    def update_epsilon(self):
        print(f"Old = Epsilon: {self.epsilon}")
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        print(f"New = Epsilon: {self.epsilon}")


def run(epochs=10, batch_size=32):
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human",mode=0,difficulty=0)
    agent = DQNAgent( env.action_space.n, env.observation_space.shape)
    observation, info = env.reset()
    total_rewards = [] 
    
    for epoch in range(epochs):
        score = 0
        #observation = preprocess_frame(observation).reshape(88,80)
        for it in range(2000):    
            
            action = agent.learn_action(observation.reshape(1, *env.observation_space.shape))
            next_observation, reward, terminated, _, _ = env.step(action)
            #next_observation = preprocess_frame(next_observation).reshape(88,80)
            agent.buffer(observation, action, reward, next_observation, terminated) 
            # model_inputs = {'lives':0, 'game_state':observation, 'score':score, 'reward':reward }
            # lives = 0            
            if reward > 0:
                score += reward
                print(f"====================")
                print(f"reward: {reward}")
                print(f"score: {score}")
                print(f"Iterations: {it}")
            if terminated:
                agent.learn_model()
                break  
            observation = next_observation


        agent.update_epsilon()
        total_rewards.append(score)

        print(f"Epoch {epoch + 1}, Total Reward: {score}")

        observation, info = env.reset()
        
    plt.plot(range(1, epochs + 1), total_rewards, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Epoch')
    plt.show()
    
    env.close()

if __name__=='__main__':
    run(epochs=20, batch_size=32)