import gymnasium as gym
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def run():
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human",mode=0,difficulty=0)
    observation, info = env.reset()

    # POSSIBLE ACTIONS:
    # 0 NONE
    # 1 SHOOT
    # 2 MOVE RIGHT
    # 3 MOVE LEFT
    # 4 MOVE RIGHT AND FIRE
    # 5 MOVE LEFT AND FIRE

    # The screen is 210 pixels wide by 160 pixels tall, with each pixel
    # having 3 different RGB values (Shape of 210,160,3)

    # can also set observation given to the AI as ram, which is the 128 byte long
    # memory of values 0 -> 255 in a 128 length array, not sure how that 
    # would perform
    prev_reward = 0
    score = 0
    for i in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        # print(action)
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
    run()