import os, numpy as np
import gymnasium as gym
from gymnasium import spaces
import crafter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback

# ---------- Crafter -> Gymnasium adapter ----------
class CrafterGymnasiumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    def __init__(self, *, reward=True, length=200, logdir=None, save_stats=True):
        base = crafter.Env(reward=reward, length=length)
        if logdir is not None:
            base = crafter.Recorder(base, logdir, save_stats=save_stats,
                                    save_video=False, save_episode=False)
        self._env, self._length, self._steps = base, length, 0
        self.observation_space = spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(17)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._steps = 0
        obs = self._env.reset()
        return obs, {}

    def step(self, action):
        self._steps += 1
        obs, reward, done, info = self._env.step(action)
        terminated = bool(done and self._steps < self._length)   # death
        truncated  = bool(done and self._steps >= self._length)  # time cap
        if done:
            self._steps = 0
        return obs, float(reward), terminated, truncated, info

    def render(self): return self._env.render()
    def close(self):  self._env.close()


# ---------- Factory ----------
EPS_LENGTH   = 200
LOG_ROOT     = "./logs"
TB_ROOT      = "./ppo_tensorboard"
RUN_NAME     = "ppo_crafter_large"

def make_env(log_subdir="ppo_train"):
    def _f():
        return CrafterGymnasiumEnv(
            reward=True,
            length=EPS_LENGTH,
            logdir=os.path.join(LOG_ROOT, log_subdir)
        )
    return _f


# ---------- Optional: simple tqdm-like progress callback ----------
class TqdmProgressCallback(BaseCallback):
    def __init__(self, total_timesteps:int):
        super().__init__()
        self.total = total_timesteps
        self._last = 0

    def _on_training_start(self):
        print(f"Training for {self.total:,} steps...")

    def _on_step(self) -> bool:
        # Print every ~10k steps (adjust as you like)
        if self.num_timesteps - self._last >= 10_000 or self.num_timesteps == self.total:
            pct = 100.0 * self.num_timesteps / self.total
            print(f"[{self.num_timesteps:,}/{self.total:,}] {pct:5.1f}%")
            self._last = self.num_timesteps
        return True


# ---------- Build envs ----------
env = DummyVecEnv([make_env("ppo_train")])   # single env, no multiprocessing
env = VecMonitor(env)                        # enables rollout/ep_* in TensorBoard

eval_env = DummyVecEnv([make_env("ppo_eval")])
eval_env = VecMonitor(eval_env)

# ---------- Model ----------
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=TB_ROOT,  # TensorBoard base dir
    # (optional) PPO knobs; defaults are fine to start
    # n_steps=2048, batch_size=64, n_epochs=10, learning_rate=2.5e-4, clip_range=0.2
)

# ---------- Callbacks ----------
TOTAL_STEPS = 100_000        # try 200_000 first if you want shorter
EVAL_FREQ   = 50_000         # run eval every N training steps
SAVE_FREQ   = 50_000         # save checkpoints every N steps

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./ppo_eval/best/",
    log_path="./ppo_eval/",
    eval_freq=EVAL_FREQ,
    deterministic=True
)
ckpt_cb = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path="./ppo_checkpoints/",
    name_prefix="ppo_crafter"
)
prog_cb = TqdmProgressCallback(total_timesteps=TOTAL_STEPS)

# ---------- Train (single call), then close ----------
model.learn(
    total_timesteps=TOTAL_STEPS,
    tb_log_name=RUN_NAME,
    log_interval=10,              # prints SB3 logs every few rollouts
    callback=[eval_cb, ckpt_cb, prog_cb]
)

env.close()
eval_env.close()

# (Optional) save final model
model.save("./ppo_final.zip")













# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# import crafter
# import os
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
# from stable_baselines3.common.callbacks import EvalCallback

# class CrafterGymnasiumEnv(gym.Env):
#     metadata = {"render_modes": ["rgb_array"]}

#     def __init__(self, *, reward=True, length=200, logdir=None, save_stats=True):
#         base = crafter.Env(reward=reward, length=length)
#         if logdir is not None:
#             base = crafter.Recorder(
#                 base, logdir, save_stats=save_stats,
#                 save_video=False, save_episode=False
#             )
#         self._env = base
#         self._length = length
#         self._steps = 0

#         # Crafter uses 64x64x3 uint8 frames and 17 discrete actions
#         self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
#         self.action_space = spaces.Discrete(17)

#     def reset(self, *, seed=None, options=None):
#         if seed is not None:
#             # Crafter doesn't expose seed(), so at least seed numpy for stochastic bits
#             np.random.seed(seed)
#         self._steps = 0
#         obs = self._env.reset()
#         return obs, {}

#     def step(self, action):
#         self._steps += 1
#         obs, reward, done, info = self._env.step(action)

#         # If done happened before hitting the cap -> terminated (e.g., death)
#         # If we hit the cap exactly -> truncated
#         terminated = bool(done and self._steps < self._length)
#         truncated = bool(done and self._steps >= self._length)
#         if done:
#             self._steps = 0
#         return obs, float(reward), terminated, truncated, info

#     def render(self):
#         return self._env.render()

#     def close(self):
#         self._env.close()


# EPS_LENGTH = 200
# NUM_EPS = 5

# def make_env():
#     def _f():
#         return CrafterGymnasiumEnv(reward=True, length=EPS_LENGTH, logdir="./logs/ppo_single")
#     return _f

# env = DummyVecEnv([make_env()])  # no subprocess -> avoids spawn/EOF issues
# env = VecMonitor(env) 

# model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
# model.learn(total_timesteps=EPS_LENGTH * NUM_EPS)
# env.close()

# eval_env = DummyVecEnv([make_env()])     # 1 env is fine for eval
# eval_env = VecMonitor(eval_env)

# eval_cb = EvalCallback(eval_env,
#                        best_model_save_path="./ppo_eval/",
#                        log_path="./ppo_eval/",
#                        eval_freq=10_000,     # run eval every N training steps
#                        deterministic=True)

# model.learn(total_timesteps=EPS_LENGTH * NUM_EPS,
#             tb_log_name="ppo_crafter",
#             callback=eval_cb)

# RUN tensorboard --logdir ./ppo_tensorboard/ after training to visualize rewards!!
# You just copy paste the localhost link into a web browser ot see the rewards
# Don't forget to pip install tensorboard if you haven't done so already!








# FIRST ATTEMPT WITH SUBPROCESS PPO
# import gym, crafter
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.vec_env import SubprocVecEnv

# EPS_LENGTH=200
# NUM_EPS=5
# NUM_ENVS=1 # PARALLELiZE TRAINING WITH MULTIPLE ENVIRONMENTS. I'd recommend 4

# def make_env():
#   def _f():
#     env = crafter.Env(reward=True, length=EPS_LENGTH)
#     return crafter.Recorder(env, "./logs/ppo_short", save_stats=True, save_video=False, save_episode=False)
#   return _f

# # env = SubprocVecEnv([make_env() for _ in range(1)])
# env = DummyVecEnv([make_env()])
# model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
# model.learn(total_timesteps=EPS_LENGTH * NUM_EPs)   # short run; not aiming for leaderboard parity
# env.close()



# DUMMYVEC IMPL
# import os
# import crafter
# import gym  # or gymnasium as gym if you're using the gymnasium wrapper
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv

# EPS_LENGTH = 200
# NUM_EPS = 5

# def make_env():
#     def _f():
#         env = crafter.Env(reward=True, length=EPS_LENGTH)
#         # use a fixed logdir for single process
#         return crafter.Recorder(env, "./logs/ppo_short", save_stats=True, save_video=False, save_episode=False)
#     return _f

# env = DummyVecEnv([make_env()])  # single env, no multiprocessing
# model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
# model.learn(total_timesteps=EPS_LENGTH * NUM_EPS)
# env.close()

