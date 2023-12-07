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
import time
import os
np.set_printoptions(threshold=sys.maxsize)


class DQNAgent:
    def __init__(self, actions, states, epsilon,epsilon_decay,epsilon_min,gamma,model=None):
        self.actions = actions
        self.states = states
        self.memory = deque(maxlen=50000)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        if model is None:
          self.model = self.build_model()
        else:
          self.model = model


    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(self.states)))
        model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
        model.add(Convolution2D(64, (3,3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.actions, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse')
        return model

    def buffer(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

# Exploration-exploitation strategy to choose an action based on epsilon-greedy
    def learn_action(self, state):
        if random.uniform(0,1) <= self.epsilon:
            return np.random.choice(self.actions)

        else:
            value = self.model.predict(state,verbose=0)
            return np.argmax(value[0]) # highest Q-value

# Learning from a minibatch of using the Q-learning
    def learn_model(self, batch_size=128):
        minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))

        states = []
        target_fs = []

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state.reshape(1, *self.states),verbose=0))

            target_f = self.model.predict(state.reshape(1, *self.states),verbose=0)
            target_f[0][action] = target
            states.append(state)
            target_fs.append(target_f)

        states = np.array(states)
        target_fs = np.array(target_fs)
        # print()
        history = self.model.fit(x=states, y=target_fs, epochs=1, verbose=0)
        # print()Z
        loss = history.history['loss'][0]
        #print(f"Loss: {loss}")

    def update_epsilon(self):
        # print(f"Old = Epsilon: {self.epsilon}")
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        # print(f"New = Epsilon: {self.epsilon}")



def preprocess_frame_grayscale(obs):

    r, g, b = obs[:,:,0],obs[:,:,1],obs[:,:,2]
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b

    grayscale /= 255
    return grayscale

def preprocess_frame_downscale(obs):

    r, g, b = obs[:,:,0],obs[:,:,1],obs[:,:,2]
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
    grayscale = grayscale.reshape(1,210,160,1)
    grayscale /= 255

    grayscale = tf.image.resize(grayscale,[105,80])[0,...,0].numpy()
    # print(grayscale.shape)
    return grayscale

def preprocess_frame_normalize(obs):

    # r, g, b = obs[:,:,0],obs[:,:,1],obs[:,:,2]
    # grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # grayscale /= 255
    return obs/255

def stack_frames(stacked_frames,state,is_new):
  frame = preprocess_frame_downscale(state)

  if is_new:
    stacked_frames = deque([np.zeros((105,80,1),dtype=np.int) for i in range(len(stacked_frames))],maxlen=4)

    for i in range(4):
      stacked_frames.append(frame)
    stacked_state = np.stack(stacked_frames,axis=0)

  else:
    stacked_frames.append(frame)
    stacked_state = np.stack(stacked_frames,axis=0)
  # print(stacked_state.shape)
  return stacked_state,stacked_frames

def run(epochs=10, batch_size=32,epsilon=0,epsilon_decay=0,epsilon_min=0,gamma=0):
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array",mode=0,difficulty=0)
    # print(env.observation_space.shape)
    # with sess:
    agent = DQNAgent( env.action_space.n, (105,80,1),epsilon,epsilon_decay,epsilon_min,gamma)
    observation, info = env.reset()
    total_rewards = []
    frames = 0
    for epoch in range(epochs):
        score = 0
        observation = preprocess_frame_downscale(observation)
        for it in range(1000):
            frames += 1
            # action = agent.learn_action(observation)
            action = agent.learn_action(observation.reshape(1,105,80,1))
            next_observation, reward, terminated, _, _ = env.step(action)
            next_observation = preprocess_frame_downscale(next_observation)

            agent.buffer(observation, action, reward, next_observation, terminated)
            # model_inputs = {'lives':0, 'game_state':observation, 'score':score, 'reward':reward }
            # lives = 0
            if reward > 0:
                score += reward
                # print(f"====================")
                # print(f"reward: {reward}")
                # print(f"score: {score}")
                # print(f"Iterations: {it}")
            if terminated:
                agent.learn_model()
                break
            observation = next_observation


        agent.update_epsilon()
        total_rewards.append(score)

        print(f"Epoch {epoch + 1}, Total Reward: {score}, Total Frames so far: {frames}")

        observation, info = env.reset()

    plt.plot(range(1, epochs + 1), total_rewards, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Epoch')
    plt.show()

    env.close()



#########################




if __name__=='__main__':    
    epsilon = 0.99
    epsilon_decay = 0.95
    epsilon_min = 0.2
    gamma = 0.2
    epochs = 100
    print("Making agent and environment...")
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human",mode=0,difficulty=0)
    agent = DQNAgent( env.action_space.n, (105,80,1),epsilon,epsilon_decay,epsilon_min,gamma)

    checkpoint = tf.train.Checkpoint(model=agent.model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="",max_to_keep=5)
    status = checkpoint.restore(manager.latest_checkpoint)
    if status:
        print("It worked")
    path = os.getcwd()+"\ckpt-200"
    checkpoint.restore(path)
    print(path)
    model = agent.model

    # env = gym.make("ALE/SpaceInvaders-v5", render_mode="human",mode=0,difficulty=0)
    # agent = DQNAgent( env.action_space.n, (105,80,1),epsilon,epsilon_decay,epsilon_min,gamma)
    print(status)
    print(model)
    observation, info = env.reset()
    total_rewards5 = []
    frames = 0
    print(model)
    for epoch in range(10):
        score = 0
        observation = preprocess_frame_downscale(observation)
        for it in range(100_000):
            frames += 1
            # action = agent.learn_action(observation)
            action = np.argmax(model.predict(observation.reshape(1,105,80,1),verbose=0)[0]) # highest Q-value
            # action = agent.learn_action(observation.reshape(1,105,80,1))
            next_observation, reward, terminated, _, _ = env.step(action)
            
            next_observation = preprocess_frame_downscale(next_observation)

            agent.buffer(observation, action, reward, next_observation, terminated)
            if reward > 0:
                score += reward
                print(f"Total Score: {score}, Total Frames: {frames}")
            if terminated:
                # agent.learn_model()
                break
            observation = next_observation
        # agent.update_epsilon()
        total_rewards5.append(score)

        print(f"Epoch {epoch + 1}, Total Reward: {score}, Total Frames so far: {frames}")

        observation, info = env.reset()
    env.close()
    print(f"Average score of 10 random games: {np.mean(total_rewards5)}")
