
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import os
import gymnasium as gym
import numpy as np
import time

env = gym.make('CartPole-v1', render_mode="human")

current_dir = r'C:\projects\personal projects\RL-Projects'
log_path = os.path.join(current_dir, 'training', 'logs')

env = Monitor(env)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)

model.learn(total_timesteps=20000)

PPO_path = os.path.join(current_dir, 'training', 'saved-models', 'PPO_model_CartPole')
model.save(PPO_path)

evaluation = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(evaluation)

env.close()