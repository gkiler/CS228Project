import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten
from keras.optimizers import Adam

def build_model(input_shape, num_actions):
    model = Sequential()
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.add(Dense(24, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

env = gym.make("ALE/SpaceInvaders-v5", render_mode="human", mode=2, difficulty=0)

# Input shape depends on the representation of the state
input_shape = env.observation_space.shape  # The shape of the state (height, width, channels)
num_actions = env.action_space.n

# Build the Q-network
model = build_model(np.prod(input_shape), num_actions)  # Use np.prod to calculate the total number of elements

epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
gamma = 0.95
memory = []

for episode in range(1000):
    state = env.reset()

    total_reward = 0

    while True:
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            # Preprocess the state (if needed) and reshape if necessary
            state_flat = np.reshape(state, [1, -1])  

            # Use the processed state in the model prediction
            action = np.argmax(model.predict(state_flat))

        # Modify the following line to handle the issue
        next_state, reward, done, _, info = env.step(action)

        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        if done:
            if len(memory) > 32:
                batch = np.random.choice(len(memory), 32, replace=False)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*[memory[i] for i in batch])
                batch_states = np.vstack(batch_states)
                batch_actions = np.array(batch_actions)
                batch_rewards = np.array(batch_rewards)
                batch_next_states = np.vstack(batch_next_states)
                batch_dones = np.array(batch_dones)

                for i in range(len(batch_states)):
                    state, action, reward, next_state, done = batch_states[i], batch_actions[i], batch_rewards[i], batch_next_states[i], batch_dones[i]

                    target = reward
                    if not done:
                        target = reward + gamma * np.amax(model.predict(np.reshape(next_state, [1, -1]))[0])
                    target_f = model.predict(np.reshape(state, [1, -1]))
                    target_f[0][action] = target
                    model.fit(np.reshape(state, [1, -1]), target_f, epochs=1, verbose=0)

            epsilon *= epsilon_decay
            epsilon = max(epsilon, min_epsilon)

            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
            break

env.close()
