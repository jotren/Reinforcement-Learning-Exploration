
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import os
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode="rgb_array")

env = Monitor(env)
env = DummyVecEnv([lambda: env])

current_dir = r'C:\projects\personal projects\RL-Projects'
PPO_path = os.path.join(current_dir, 'training', 'saved-models', 'PPO_model_CartPole')

model = PPO.load(PPO_path, env=env)

obs = env.reset()
print(obs)

action, _ = model.predict(obs)

print(action)

env.step(action)
