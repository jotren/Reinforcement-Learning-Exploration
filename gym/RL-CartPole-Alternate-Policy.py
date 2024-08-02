from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import os
import gymnasium as gym
import numpy as np
import time


env = gym.make('CartPole-v1', render_mode="human")
env = Monitor(env)
env = DummyVecEnv([lambda: env])

current_dir = r'C:\projects\personal_projects\RL-Projects'

log_path = os.path.join(current_dir, 'training', 'logs')
save_path = os.path.join(current_dir, 'training', 'saved_models')

# Here we define what the reward threshold is
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=5000, best_model_save_path=save_path, verbose=1)


# Start the model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch':dict(pi=[128,128,128,128], vf=[128,128,128,128])})

model.learn(total_timesteps=20000, callback=eval_callback)