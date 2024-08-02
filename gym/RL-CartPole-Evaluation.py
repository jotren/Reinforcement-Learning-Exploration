from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import os
import gymnasium as gym


env = gym.make('CartPole-v1', render_mode="human")

env = Monitor(env)
env = DummyVecEnv([lambda: env])

current_dir = r'C:\projects\personal projects\RL-Projects'
PPO_path = os.path.join(current_dir, 'training', 'saved-models', 'PPO_model_CartPole')

model = PPO.load(PPO_path, env=env)

episodes = 5
timeSteps = 100

episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        
    print('Episode: {} Score: {}'.format(episode, score))
    
env.close()