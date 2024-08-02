import gymnasium as gym
import numpy as np
import time

env = gym.make('CartPole-v1', render_mode="human")

# Specify the timesteps 
timeSteps = 100

# This loop just runs the program and takes a random sample (left-right 0/1) to make it work.
for episodeIndex in range(timeSteps):
    
    initial_state=env.reset()
    print(episodeIndex)
    appendedObservations = []
    
    for timeIndex in range(timeSteps):
        
        random_action = env.action_space.sample()
        n_state, reward, done, boolean, info = env.step(random_action)
        appendedObservations.append(n_state)
        time.sleep(0.01)
        if (done):
            time.sleep(0.1)
            break

env.close()
