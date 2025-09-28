import gym
import crafter 
import numpy as np, os
from tqdm import tqdm
import matplotlib.pyplot as plt


EPS_LENGTH=200
NUM_EPS=1

logdir = "./logs/random"
# env = gym.make('CrafterReward-v1')
# length is how long each episode is.
env = crafter.Env(reward=True, length=EPS_LENGTH)
env = crafter.Recorder(env, logdir, save_stats=True, save_video=True, save_episode=False)

num_steps = NUM_EPS * EPS_LENGTH
done = False
rewards = []
episode_rewards = 0
obs = env.reset()
for _ in tqdm(range(num_steps)):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    episode_rewards += reward
    if done:
        print(f"Episode return: {episode_rewards}")
        episode_rewards = 0
        obs = env.reset()
env.close()

# visualize rewards (raw and smoothed)
plt.plot(rewards, alpha=0.3, label="Instant reward")
window = 50
smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
plt.plot(range(window-1, len(rewards)), smoothed, label=f"Moving avg ({window})")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.legend()
plt.show()