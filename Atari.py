import gymnasium as gym
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
np.set_printoptions(threshold=sys.maxsize)

def game_model(state_input, meta_input, action_output):

    # game state branch
    # each observation has shape of (210, 160, 3)
    state_input = tf.keras.layers.Input(shape=(state_input.shape))
    print(state_input.shape)
    state_conv1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2,2), padding="same", activation='relu')(state_input)
    print("conv1",state_conv1.shape)
    state_conv2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2,2), padding="same", activation='relu')(state_conv1)
    print("conv2",state_conv2.shape)
    state_conv3 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2,2), padding="same", activation='relu')(state_conv2)
    print("conv3",state_conv3.shape)
    state_flatten = tf.keras.layers.Flatten()(state_conv3)
    print("flattened",state_flatten.shape)


    # meta data stored in order of: lives, scores, rewards
    meta_input = tf.keras.layers.Input(shape=(meta_input.shape))
    meta_dense = tf.keras.layers.Dense(64, activation='relu')(meta_input)
    meta_reshaped = tf.keras.layers.Reshape((3* 64,))(meta_dense)
    print(meta_reshaped.shape)

    # Merge the layers
    merged = tf.keras.layers.Concatenate(axis=1)([state_flatten, meta_reshaped])
    print("merged",merged.shape)
                        
    # output: action
    output_layer = tf.keras.layers.Dense(6, activation='softmax')(merged)
    print("output",output_layer.shape)

    # Create the model
    model = tf.keras.Model(inputs=[state_input, meta_input], outputs=output_layer)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    y_train = to_categorical(action_output, num_classes=6)
    x_train = [state_input, meta_input]
    hist = model.fit(
        x_train,
        y_train, 
        epochs=80,
    )

    # # get test and train accuracy and loss
    # train_loss = hist.history["loss"]
    # train_accuracy = hist.history["categorical_accuracy"]

    return model#, train_loss, train_accuracy

def run():
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human",mode=0,difficulty=0)
    observation, info = env.reset() #  can set to a specific seed

    prev_reward = 0
    score = 0

    lives = []
    rewards = []
    scores = [] 
    game_states = []
    actions = []
    for i in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        # print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        # if reward > 0: 
        #     print(reward) 

        # print(env.observation_space)  # Box(0, 255, (210, 160, 3), uint8)
        # print(observation.shape)      # (210, 160, 3) 

        prev_reward = max(reward,prev_reward)
        score += reward
        
        lives.append(info['lives']) 
        rewards.append(reward)
        scores.append(score)
        game_states.append(observation)
        actions.append(action)

        state_inputs = np.array(game_states)
        meta_inputs  = np.array([lives, scores, rewards]) 
        action_outputs = np.array(actions) 
        print(action_outputs.shape)
        #model, train_loss, train_accuracy = game_model(state_inputs, meta_inputs, action_outputs)

        model = game_model(state_inputs, meta_inputs, action_outputs)
        # print(train_accuracy)
        # print(train_loss)

        print("---------------")
        if terminated or truncated:
            observation, info = env.reset()
            print("iteration:",i) 

    env.close()

if __name__=='__main__':
    run()