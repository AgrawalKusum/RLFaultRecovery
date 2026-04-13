import numpy as np
import os
import pickle
import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from motion_imitation.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from stable_baselines3.common.monitor import Monitor
from motion_imitation.envs import env_builder

class MasterBrain:
    def __init__(self, 
                 env,  # Added this argument
                 nominal_model, nominal_stats_path,
                 recovery_model, recovery_stats_path,
                 recovery_task,
                 sac_action_limit=0.5):
        
        def get_patched_env(builder_func):
            raw = builder_func()
            # Standard Gymnasium/Gym patching
            old_reset = raw.reset
            raw.reset = lambda **kwargs: (old_reset(**kwargs), {}) if not isinstance(old_reset(**kwargs), tuple) else old_reset(**kwargs)
            old_step = raw.step
            def patched_step(action):
                val = old_step(action)
                return (val[0], val[1], val[2], False, val[3]) if len(val) == 4 else val
            raw.step = patched_step
            return Monitor(raw)

        # --- NOMINAL SETUP (160 Dims) ---
        nom_builder = lambda: env_builder.build_imitation_env(
            motion_files=["motion_imitation/data/motions/dog_pace.txt"],
            num_parallel_envs=1, mode="test", enable_randomizer=False, enable_rendering=False)
        self.nom_env = VecNormalize.load(nominal_stats_path, DummyVecEnv([lambda: get_patched_env(nom_builder)]))

        # --- RECOVERY SETUP (84 Dims) ---
        # Note: We use your actual SAC builder here
        rec_builder = lambda: env_builder.build_sac_recovery_env(
            enable_randomizer=False, enable_rendering=False)
        self.rec_env = VecNormalize.load(recovery_stats_path, DummyVecEnv([lambda: get_patched_env(rec_builder)]))

        for e in [self.nom_env, self.rec_env]:
            e.training = False
            e.norm_reward = False

        # We use the passed 'env' to create the dummy wrappers
        #self.nom_env = VecNormalize.load(nominal_stats_path, DummyVecEnv([lambda: wrap_for_sb3(env)]))
        self.nom_env.training = False
        self.nom_env.norm_reward = False

        #self.rec_env = VecNormalize.load(recovery_stats_path, DummyVecEnv([lambda: wrap_for_recovery(env)]))
        self.rec_env.training = False
        self.rec_env.norm_reward = False

        self.nominal = nominal_model
        self.recovery = recovery_model
        self.task = recovery_task
        self.sac_scale = sac_action_limit

        # State Management
        self.state = "NOMINAL"
        self.timer = 0.0
        
        self.cfg = {
            "panic_tilt": 0.5,
            "safe_tilt": 0.15,
            "panic_height": 0.22,
            "safe_height": 0.30,
            "settle_time": 0.5,
            "panic_debounce": 0.03,
        }
    def _flatten_obs(self, obs_dict):
        # The order must match the 'sensors' list in your env_builder.py
        # Historically: MotorAngle -> IMU -> LastAction
        flat_obs = []
        for key in ['motor_angle', 'imu', 'last_action']:
            if key in obs_dict:
                flat_obs.append(np.array(obs_dict[key]).flatten())
        
        return np.concatenate(flat_obs)
    def _load_stats(self, path):
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _get_robot_state(self, env):
        """Reaches through wrappers to pull raw physics data."""
        actual_env = env.unwrapped
        roll, pitch, _ = actual_env.robot.GetTrueBaseRollPitchYaw()
        height = actual_env.robot.GetBasePosition()[2]
        tilt = np.sqrt(roll**2 + pitch**2)
        return tilt, height

    def _normalize_obs(self, obs, stats):
        """Manually applies VecNormalize stats to the observation."""
        if stats is None:
            return obs
        # VecNormalize formula: (obs - mean) / sqrt(var + epsilon)
        obs_mean = stats.obs_rms.mean
        obs_var = stats.obs_rms.var
        epsilon = 1e-8
        return (obs - obs_mean) / np.sqrt(obs_var + epsilon)

    def get_action(self, env, observation):
        if isinstance(observation, tuple):
            observation = observation[0]

        if isinstance(observation, dict):
            observation = self._flatten_obs(observation)

        observation = np.array(observation).reshape(1, -1)
        
        tilt, height = self._get_robot_state(env)
        dt = env.env_time_step # 0.01s for Laikago (100Hz)

        # --- HYSTERESIS SWITCHING LOGIC ---
        if self.state == "NOMINAL":
            # Check for Fall/Deflection
            if tilt > self.cfg["panic_tilt"] or height < self.cfg["panic_height"]:
                self.timer += dt
                if self.timer >= self.cfg["panic_debounce"]:
                    print(f"\n[MASTER] PANIC SWITCH -> RECOVERY (Tilt: {round(tilt,2)})")
                    self.state = "RECOVERY"
                    self.timer = 0.0
            else:
                self.timer = 0.0

        elif self.state == "RECOVERY":
            # Check for Recovery Success
            is_upright = tilt < self.cfg["safe_tilt"]
            is_high = height > self.cfg["safe_height"]
            
            if is_upright and is_high:
                self.timer += dt
                if self.timer >= self.cfg["settle_time"]:
                    print(f"\n[MASTER] SUCCESS HANDOFF -> NOMINAL (Height: {round(height,2)})")
                    self.state = "NOMINAL"
                    self.timer = 0.0
            else:
                self.timer = 0.0

        # --- BRAIN EXECUTION ---
        if self.state == "NOMINAL":
            # Normalize and predict
            norm_obs = self.nom_env.normalize_obs(observation)
            action, _ = self.nominal.predict(norm_obs, deterministic=True)
            return action[0] # Return the first (and only) action in the batch
        
        else:
            # --- RECOVERY SLICING ---
            # Slicing the second dimension of our (1, 160) array to get (1, 84)
            rec_obs = observation[:, :84] 
            
            norm_obs = self.rec_env.normalize_obs(rec_obs)
            action, _ = self.recovery.predict(norm_obs, deterministic=True)
            return action[0] * self.sac_scale