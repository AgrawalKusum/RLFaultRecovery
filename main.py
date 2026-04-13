import os
import time
import numpy as np
from stable_baselines3 import PPO, SAC
from motion_imitation.envs import env_builder
from motion_imitation.robots import robot_config
from MasterBrain import MasterBrain
from disturbances import DisturbanceManager

def main():
    # --- 1. CONFIGURATION ---
    NOMINAL_MODEL_PATH = "models/final_model.zip"
    NOMINAL_STATS_PATH = "models/vec_normalize.pkl"
    RECOVERY_MODEL_PATH = "output_new5/final_model.zip"
    RECOVERY_STATS_PATH = "output_new5/vec_normalize.pkl"
    MOTION_FILE = "motion_imitation/data/motions/dog_pace.txt"
    # --- 2. ENVIRONMENT SETUP ---
    # We use build_laikago_env as it provides the standard walking setup
    env = env_builder.build_imitation_env(
    motion_files=[MOTION_FILE],
    num_parallel_envs=1,
    mode="test",
    enable_randomizer=True,
    enable_rendering=True
    )
    
    pusher = DisturbanceManager(env)

    # --- 3. LOAD BRAINS ---
    print("Loading models...")
    # Load weights
    nom_model = PPO.load(NOMINAL_MODEL_PATH)
    rec_model = SAC.load(RECOVERY_MODEL_PATH)
    
    # Initialize the Supervisor
    # Note: sac_action_limit should match your training's SimpleRobotOffsetGenerator
    master = MasterBrain(
        env=env,
        nominal_model=nom_model,
        nominal_stats_path=NOMINAL_STATS_PATH,
        recovery_model=rec_model,
        recovery_stats_path=RECOVERY_STATS_PATH,
        recovery_task=env.task,
        sac_action_limit=0.5 
    )

    # --- 4. THE EXECUTION LOOP ---
    obs, _ = env.reset()
    step_count = 0
    
    print("\n--- Starting Simulation ---")
    print("Robot is WALKING. Use Ctrl+C to stop.")
    
    try:
        while True:
            # A. Apply a push every 800 steps (8 seconds)
            if step_count > 0 and step_count % 800 == 0:
                # 600N is usually enough to knock over a Laikago/A1
                pusher.apply_push(force_magnitude=650, direction='lateral')

            # B. Supervisor selects the correct brain
            action = master.get_action(env, obs)

            # C. Step the world
            obs, reward, terminated, truncated,info = env.step(action)
            done = terminated or truncated
            # D. Logging
            if step_count % 50 == 0:
                print(f"Mode: {master.state:8} | Steps: {step_count:5}", end="\r")

            step_count += 1
            
            if done:
                print("\n[INFO] Environment reset.")
                obs = env.reset()
                master.state = "NOMINAL"
                step_count = 0
                
            # Keep sim timing steady for visualization
            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\nShutting down...")
        env.close()

if __name__ == "__main__":
    main()