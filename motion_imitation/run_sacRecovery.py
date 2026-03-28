import os
import inspect
import argparse
import numpy as np
import multiprocessing as mp
import gymnasium as gym
import tqdm
# Path setup to match your PPO script
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from motion_imitation.envs import env_builder
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# --- The Curriculum Callback (Handles multi-proc syncing) ---
class SACCurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SACCurriculumCallback, self).__init__(verbose)
        self.steps_at_level = 0

    def _on_step(self) -> bool:
        self.steps_at_level += 1
        # Check infos from ALL parallel processes
        for info in self.locals.get("infos", []):
            if "episode" in info:
                # In your task, 'done' only triggers on success
                success = info.get("is_success", False)
                # Tell all sub-processes to update their curriculum
                level_up = self.training_env.env_method('update_curriculum', success, self.steps_at_level)
                if any(level_up):
                    self.steps_at_level = 0
        return True

class SB3MultiCorePatch(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Explicitly cast the spaces to Gymnasium Box spaces
        self.observation_space = gym.spaces.Box(
            low=np.array(env.observation_space.low), 
            high=np.array(env.observation_space.high), 
            shape=env.observation_space.shape, 
            dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array(env.action_space.low), 
            high=np.array(env.action_space.high), 
            shape=env.action_space.shape, 
            dtype=np.float32)
        
    def __getattr__(self, name):
        """
        The 'Master Key': If a method isn't found in the wrapper, 
        check if it exists in the Task object.
        """
        if name == '_task': # Prevent infinite recursion
            return getattr(self.env, name)
            
        # Try to find the attribute in the underlying environment's task
        try:
            task = getattr(self.env.unwrapped, '_task')
            return getattr(task, name)
        except AttributeError:
            # Fallback to the standard behavior
            return super().__getattribute__(name)

    def reset(self, **kwargs):
        # Handle the transition from old gym reset to Gymnasium reset (obs, info)
        result = self.env.reset() 
        
        # Standard Gymnasium reset expects (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def step(self, action):
        # Handle the transition to (obs, rew, terminated, truncated, info)
        result = self.env.step(action)
        if isinstance(result, tuple):
            if len(result) == 4: 
                obs, rew, done, info = result
                # Gymnasium expects 5 values (obs, rew, terminated, truncated, info)
                return obs, rew, done, False, info
            if len(result) == 5: 
                return result
        return result

class RecoveryCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose: int = 1):
        super(RecoveryCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        # Print stats every 'check_freq' steps
        if self.n_calls % self.check_freq == 0:
            # Access the reward from the last step
            # Note: with SubprocVecEnv, we pull from the info dict
            rewards = self.locals.get("rewards")
            if rewards is not None:
                mean_reward = np.mean(rewards)
                print(f"Step: {self.num_timesteps} | Mean Reward: {mean_reward:.3f}")
        return True
    
def make_env(args):
    def _init():
        # Build raw environment
        raw_env = env_builder.build_sac_recovery_env(
            enable_randomizer=True,
            enable_rendering=(args.mode == "test" and args.visualize)
        )
        # Note: We don't need SB3MultiCorePatch if build_sac_recovery_env 
        # already returns Gymnasium-compatible spaces.
        env = SB3MultiCorePatch(raw_env)

        return Monitor(env)
    return _init

def train(model, env, total_timesteps, output_dir, int_save_freq):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    callbacks = [SACCurriculumCallback(),
                 RecoveryCallback(check_freq=1000)]
    if int_save_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=int_save_freq, 
            save_path=os.path.join(output_dir, "intermediate"),
            name_prefix='model',
            save_vecnormalize=True,
            save_replay_buffer=True
        ))

    print(f"Starting SAC training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callbacks, progress_bar=True)
    
    model.save(os.path.join(output_dir, "final_model"))
    env.save(os.path.join(output_dir, "vec_normalize.pkl"))

def test(model, env, num_procs, num_episodes=None):
  curr_return = np.zeros(num_procs)
  sum_return = 0
  episode_count = 0
  success_count = 0

  if num_episodes is not None:
    num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
  else:
    num_local_episodes = 20 # Default for quick check

  print(f"Starting SAC Test: {num_episodes} total episodes across {num_procs} procs...")
  
  obs = env.reset()
  while episode_count < num_local_episodes:
    # THE INFERENCE STEP
    action, _ = model.predict(obs, deterministic=True)
    
    # THE ENVIRONMENT STEP
    obs, rewards, dones, infos = env.step(action)
    
    curr_return += rewards

    for i in range(num_procs):
        if dones[i]:
            sum_return += curr_return[i]
            curr_return[i] = 0
            episode_count += 1
            
            # Check if this episode was a 'Success' (Standing Up)
            # In your SAC task, 'done' only returns True if standing and pose_good
            success_count += 1 
            
            if episode_count >= num_local_episodes:
                break

  # Aggregate across MPI if you are using it, otherwise standard mean
  mean_return = sum_return / episode_count if episode_count > 0 else 0
  success_rate = (success_count / episode_count) * 100 if episode_count > 0 else 0
  
  print("-" * 45)
  print(f"SAC TEST RESULTS")
  print(f"Episodes: {episode_count} | Success Rate: {success_rate:.1f}%")
  print(f"Mean Return: {mean_return:.2f}")
  print("-" * 45)

  return

def main():
  mp.set_start_method('forkserver', force=True)
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--mode", type=str, default="train")
  arg_parser.add_argument("--visualize", action="store_true", default=False)
  arg_parser.add_argument("--output_dir", type=str, default="output_sac")
  arg_parser.add_argument("--num_test_episodes", type=int, default=20)
  arg_parser.add_argument("--model_file", type=str, default="")
  arg_parser.add_argument("--total_timesteps", type=int, default=30000000)
  arg_parser.add_argument("--int_save_freq", type=int, default=500000)

  args = arg_parser.parse_args()
  
  num_procs = 8
  os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

  # 1. Setup Environment
  env = SubprocVecEnv([make_env(args) for _ in range(num_procs)])

  # 2. Initialization / Resume Logic
  if args.model_file != "":
    # Load Stats
    stats_path = os.path.join(os.path.dirname(args.model_file), "vec_normalize.pkl")
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = (args.mode == "train")
        env.norm_reward = (args.mode == "train")
        print(f"Loaded Normalization Stats: {stats_path}")
    
    # Load Weights
    model = SAC.load(args.model_file, env=env)
    print(f"Resuming SAC Model: {args.model_file}")
  else:
    print("Starting SAC Training from Scratch...")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=args.output_dir,
                policy_kwargs=dict(net_arch=[512, 256]), 
                buffer_size=1000000, batch_size=256, device="cpu")

  # 3. Mode Switch
  if args.mode == "train":
      train(model=model, env=env, total_timesteps=args.total_timesteps,
            output_dir=args.output_dir, int_save_freq=args.int_save_freq)
  elif args.mode == "test":
      env.training = False
      env.norm_reward = False
      test(model=model, env=env, num_procs=num_procs, num_episodes=args.num_test_episodes)

  return

if __name__ == '__main__':
    main()