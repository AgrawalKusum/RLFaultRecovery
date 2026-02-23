# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["IN_SUBPROC"] = "1"
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import gymnasium as gym
from gymnasium.wrappers.compatibility import EnvCompatibility
import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import time
import torch
from motion_imitation.envs import env_builder as env_builder
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.learning import ppo_imitation as ppo_imitation

from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 512

ENABLE_ENV_RANDOMIZER = True

def set_rand_seed(seed=None):
  if seed is None:
        seed = int(time.time())
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  return

def build_model(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
  policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[512, 256], vf=[512, 256])
    )

  timesteps_per_actorbatch = int(np.ceil(float(timesteps_per_actorbatch) / num_procs))
  optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

  model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-5,
        n_steps=4096, 
        device="cpu",      # This replaces timesteps_per_actorbatch
        batch_size=256,
        n_epochs=1,
        gamma=0.95,
        gae_lambda=0.95,
        policy_kwargs=policy_kwargs,
        tensorboard_log=output_dir,
        verbose=1
    )
  return model


def train(model, env, total_timesteps, output_dir="", int_save_freq=0):
  # if (output_dir == ""):
  #   save_path = None
  # else:
  #   save_path = os.path.join(output_dir, "model.zip")
  #   if not os.path.exists(output_dir):
  #     os.makedirs(output_dir)
  

  # callbacks = []
  # # Save a checkpoint every n steps
  # if (output_dir != ""):
  #   if (int_save_freq > 0):
  #     int_dir = os.path.join(output_dir, "intermedate")
  #     callbacks.append(CheckpointCallback(save_freq=int_save_freq, save_path=int_dir,
  #                                         name_prefix='model'))

  # model.learn(total_timesteps=total_timesteps, save_path=save_path, callback=callbacks)

  # return
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  callbacks = []
  
  # 1. Add the Progress Bar Callback
  callbacks.append(ProgressBarCallback())

  # 2. Add Intermediate Checkpoints
  if (output_dir != "" and int_save_freq > 0):
    int_dir = os.path.join(output_dir, "intermediate")
    # save_vecnormalize=True ensures stats are saved alongside weights
    callbacks.append(CheckpointCallback(save_freq=int_save_freq, 
                                        save_path=int_dir,
                                        name_prefix='model',
                                        save_vecnormalize=True))

  # 3. Start Learning (Note: save_path is removed as SB3 learn() doesn't use it)
  print(f"Starting training for {total_timesteps} steps...")
  model.learn(total_timesteps=total_timesteps, callback=callbacks)

  # 4. Final Save (Weights + Stats)
  final_model_path = os.path.join(output_dir, "final_model.zip")
  model.save(final_model_path)
  
  if hasattr(env, "save"):
    stats_path = os.path.join(output_dir, "vec_normalize.pkl")
    env.save(stats_path)
    print(f"Environment stats saved to {stats_path}")

  print(f"Training complete. Final model saved to {final_model_path}")
  return

def test(model, env, num_procs, num_episodes=None):
  curr_return = 0
  sum_return = 0
  episode_count = 0

  if num_episodes is not None:
    num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
  else:
    num_local_episodes = np.inf

  o,_ = env.reset()
  while episode_count < num_local_episodes:
    a, _ = model.predict(o, deterministic=True)
    #o, r, done, info = env.step(a)
    #curr_return += r
    o, r, terminated, truncated, info = env.step(a) 
    done = terminated or truncated # Combine for the loop logic
    curr_return += r[0]

    if terminated[0] or truncated[0]:
      sum_return += curr_return
      curr_return = 0
      episode_count += 1

  sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
  episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

  if episode_count > 0:
      mean_return = sum_return / episode_count
  else:
      mean_return = 0

  if MPI.COMM_WORLD.Get_rank() == 0:
      print("Mean Return: " + str(mean_return))
      print("Episode Count: " + str(episode_count))

  return

def make_env(args, enable_env_rand):
    def _init():
        # Build raw environment (NO GUI for speed)
        raw_env = env_builder.build_imitation_env(
            motion_files=[args.motion_file],
            num_parallel_envs=1,
            mode=args.mode,
            enable_randomizer=enable_env_rand,
            enable_rendering=False)
        
        # Wrap for Gymnasium compatibility
        from gymnasium.wrappers.compatibility import EnvCompatibility
        env = EnvCompatibility(raw_env)

        # Explicitly cast spaces to Gymnasium
        env.observation_space = gym.spaces.Box(
            low=env.observation_space.low, 
            high=env.observation_space.high, 
            shape=env.observation_space.shape, 
            dtype=env.observation_space.dtype)
        env.action_space = gym.spaces.Box(
            low=env.action_space.low, 
            high=env.action_space.high, 
            shape=env.action_space.shape, 
            dtype=env.action_space.dtype)
        
        # Monitor tracks rewards/lengths for logs
        return Monitor(env)
    return _init

import multiprocessing as mp

def main():
  mp.set_start_method('forkserver', force=True)
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
  arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
  arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/dog_pace.txt")
  arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
  arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
  arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None)
  arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
  arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
  arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int, default=0) # save intermediate model every n policy steps

  args = arg_parser.parse_args()
  
  num_procs = 8
  os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
  
  enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")

  env = SubprocVecEnv([make_env(args, enable_env_rand) for _ in range(num_procs)])
    
  # Add Normalization (Standard for PPO walking)
  env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
 
  model = build_model(env=env,
                      num_procs=num_procs,
                      timesteps_per_actorbatch=TIMESTEPS_PER_ACTORBATCH,
                      optim_batchsize=OPTIM_BATCHSIZE,
                      output_dir=args.output_dir)

  
  if args.model_file != "":
    print(f"Resuming training from: {args.model_file}")
    
    # 1. Load the model weights into the existing env
    model = PPO.load(args.model_file, env=env)

    # 2. Look for the normalization statistics (.pkl file)
    # We assume it's in the same folder as the model
    #stats_path = os.path.join(os.path.dirname(args.model_file), "vecnormalize.pkl")
    stats_path = "vec_normalize.pkl"

    if os.path.exists(stats_path):
        # Wrap the current env with the saved stats
        # 'training=True' allows the stats to keep updating during the next 195M steps
        env = VecNormalize.load(stats_path, env)
        env.training = True 
        model.set_env(env)
        print(f"Successfully loaded normalization stats from {stats_path}")
    else:
        print("Warning: No normalization stats found. Training might be unstable!")

  if args.mode == "train":
      train(model=model, 
            env=env, 
            total_timesteps=args.total_timesteps,
            output_dir=args.output_dir,
            int_save_freq=args.int_save_freq)
  elif args.mode == "test":
      test(model=model,
           env=env,
           num_procs=num_procs,
           num_episodes=args.num_test_episodes)
  else:
      assert False, "Unsupported mode: " + args.mode

  return

if __name__ == '__main__':
  main()
