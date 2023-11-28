import gymnasium as gym
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten
from keras.optimizers import Adam

np.set_printoptions(threshold=sys.maxsize)

class DQNAgent:
    def __init__(self, height, width, channels, actions):
        self.height = height
        self.width = width
        self.channels = channels
        self.actions = actions
        self.model = self.build_model(height, width, channels, actions)
        

    def build_model(self, height, width, channels, actions):
        model = Sequential()
        model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(height, width, channels)))
        model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
        model.add(Convolution2D(64, (3,3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(actions, activation='linear'))
        model.compile(optimizer=Adam(lr=0.0001), loss='mse')
        return model
    def learn_action(self, state):
        value = self.model.predict(state.reshape((1, self.height, self.width, self.channels)))
        action = np.argmax(value) 
        return action
        
    # end def

def run(epochs=10):
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human",mode=0,difficulty=0)
    height, width, channels = env.observation_space.shape
    agent = DQNAgent(height, width, channels, env.action_space.n)
    observation, info = env.reset()

    for epoch in range(epochs):
        prev_reward = 0
        score = 0
        for _ in range(1000):    
       
            action = agent.learn_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            
            prev_reward = max(reward,prev_reward)
            score += reward
            
            lives = info['lives']
            curr_game_state = observation

            model_inputs = {'lives':info['lives'],
                            'game_state':observation,
                            'score':score,
                            'reward':reward
                            }
            # print(model_inputs)
            # print(np.array(observation).shape)
            # for val in observation:
            #     print(val,end=", ")
            # print()
            if terminated or truncated:
                observation, info = env.reset()

    env.close()

if __name__=='__main__':
    run(epochs=10)