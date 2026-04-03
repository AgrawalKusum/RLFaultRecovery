import numpy as np
import random
import json
import os

class SACRecoveryTask(object):
    def __init__(self, 
                 target_pose=None, 
                 fall_angle_threshold=0.5,
                 done_after_standing=True,
                 time_penalty=-0.05):
        """
        Args:
            target_pose: The joint angles (12,) the robot should reach to 'handover'.
            fall_angle_threshold: Radian limit for roll/pitch before it's a 'fall'.
            done_after_standing: If True, episode ends once stabilized.
        """
        self._done_after_standing = done_after_standing
        self._target_pose = target_pose
        self._time_penalty = time_penalty
        self._curriculum_level = 0.1  # Starts at 10% of max chaos
        self._success_buffer = []      # Track last 100 episodes
        self.last_success = False     # Store last success for curriculum update

        self._standing_reward = 0.0 # Bonus for successfully standing up (encourages SAC to find it)
        self._prev_height = None # For progress reward

    def __call__(self, env):
        """This makes the task object 'callable' like a function."""
        return self.reward(env)
    
    def _get_chaos_limits(self):
        """Linearly scale the chaos based on the current level (0.0 to 1.0)."""
        # Level 0.1: ~18 degrees tilt
        # Level 1.0: 180 degrees (full flip)
        max_angle = self._curriculum_level * np.pi
        return max_angle

    def reset(self, env):
        """Standard task reset called by LocomotionGymEnv."""
        self._env = env
        self._prev_height = None # Reset height tracking for progress reward
        self._standing_reward = 0.0 # Reset standing reward

    def custom_reset(self, env):
        """
        The 'Chaos' Reset: Randomizes 3D Orientation and Joint Angles.
        This is called by your modified LocomotionGymEnv.reset().
        """
        print(f"DEBUG: Chaos Reset Triggered! Level: {self._curriculum_level}")

        if random.random() < 1:
            spawn_pos = [0, 0, 0.32] # Target height
            spawn_quat = [0, 0, 0, 1] # Perfectly upright
            joints = self._target_pose
            print("[DEBUG]: joins=", joints)
            print("DEBUG: Perfect Reset sneak peak")

            env.robot.Reset(reload_urdf=False, 
                            default_motor_angles=joints,
                            reset_time=0.0)
            env.pybullet_client.resetBasePositionAndOrientation(
                env.robot.quadruped, spawn_pos, spawn_quat
            )
        else:
            
            # 1. Randomize Orientation (Full 3D Chaos)
            level= self._curriculum_level
            angle_limit = self._get_chaos_limits()
            random_roll = random.uniform(-angle_limit, angle_limit)
            random_pitch = random.uniform(-angle_limit, angle_limit)
            random_yaw = random.uniform(-np.pi, np.pi)
            
            chaos_quat = env.pybullet_client.getQuaternionFromEuler([
                random_roll, random_pitch, random_yaw
            ])

            # 2. HEIGHT (Curriculum Controlled)
            # At 0.1: base_height is low (feet touching).
            # At 1.0: base_height is 0.8m (dropping from air).
            safe_height = 0.32 + (0.13 * np.cos(max(abs(random_roll), abs(random_pitch))))
            safe_height = max(0.32, safe_height)
            chaos_height = 0.6 + random.uniform(0, 0.2)
            
            # Linear interpolation between safe and chaos
            spawn_height = (1 - level) * safe_height + (level * chaos_height)
            random_pos = [0, 0, spawn_height] 

            # 3. JOINTS (Curriculum Controlled)
            # At 0.1: Start exactly at Target Pose (Feet ready).
            # At 1.0: Start with totally random joints (Legs tangled).
            target_pose = np.array(self._target_pose)
            random_noise = max(0, (level - 0.1) * 1.5) # Increases with level
            
            random_joints = [
                t + random.uniform(-random_noise, random_noise) 
                for t in target_pose
            ]

            # 4. Apply to Physics Engine
            # We set reset_time=0 so the 'fall' starts instantly
            env.robot.Reset(reload_urdf=False, 
                            default_motor_angles=random_joints,
                            reset_time=0.0)
            
            env.pybullet_client.resetBasePositionAndOrientation(
                env.robot.quadruped, random_pos, chaos_quat
            )

    def update_curriculum(self, success, steps_at_level):
        """
        Args:
            success: Boolean (Did it stand up?)
            steps_at_level: Int (How many steps since last level up)
        """
        self._success_buffer.append(1.0 if success else 0.0)
        if len(self._success_buffer) > 100:
            self._success_buffer.pop(0)
            
        avg_success = np.mean(self._success_buffer)

        # 1. Hysteresis Logic (70% to go up, 30% to go down)
        if avg_success > 0.7 and self._curriculum_level < 1.0:
            self._curriculum_level += 0.05
            return True # Level Up!
            
        # 2. Pity Logic (Force progression if stuck for 100k steps)
        elif steps_at_level > 100000 and self._curriculum_level < 1.0:
            self._curriculum_level += 0.02 
            print(f"Pity Progression: Level increased to {self._curriculum_level}")
            return True

        elif avg_success < 0.3 and self._curriculum_level > 0.1:
            self._curriculum_level -= 0.05
            
        return False
    
    def save_curriculum(self, path="output/curriculum_state.json"):
        """Saves the current level and success buffer to a file."""
        state = {
            "level": self._curriculum_level,
            "buffer": self._success_buffer
        }
        with open(path, 'w') as f:
            json.dump(state, f)
        print(f"[SAVE] Curriculum state saved at Level: {round(self._curriculum_level, 3)}")

    def load_curriculum(self, path="output/curriculum_state.json"):
        """Loads the curriculum level and buffer from a file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                state = json.load(f)
                self._curriculum_level = state.get("level", 0.1)
                self._success_buffer = state.get("buffer", [])
            print(f"[LOAD] Curriculum restored to Level: {round(self._curriculum_level, 3)}")
            return True
        return False

    def reward(self, env):
        """
        Multi-term reward for SAC to learn recovery.
        """
        #print("DEBUG: Reward function called")
        robot = env.robot
        # Get current state
        cur_joints = robot.GetMotorAngles()
        roll, pitch, _ = robot.GetTrueBaseRollPitchYaw()
                
        # Term 1: Pose Reward (How close are we to the 'Ready' stance?)
        # Use negative MSE between current and target joints
        pose_dist = np.mean(np.square(cur_joints - self._target_pose))
        r_pose = np.exp(-8.0 * pose_dist)

        # Term 2: Uprightness Reward (Penalty for being tilted)
        # We want roll and pitch to be near 0
        orientation_dist = np.sqrt(roll**2 + pitch**2)
        r_upright = np.exp(-3.0 * orientation_dist)

        # Term 3: Height Reward
        # Encourage the base to be at a standing height (~0.45m for Laikago)
        #base_pos, _ = robot.GetBasePositionAndOrientation()
        base_pos = robot.GetBasePosition()
        target_height = 0.32
        r_height = np.clip(base_pos[2] / target_height, 0, 1)

        # Term 4: Energy Penalty
        # Discourage spastic leg movements (helps SAC converge on smooth motions)
        r_energy = -0.002 * np.mean(np.square(robot.GetMotorVelocities()))

        #Term 5: Time Penalty
        # Encourage faster recovery by penalizing time taken
        r_time = -0.02

        #Term 6: Progress Reward
        if self._prev_height is None:
            self._prev_height = base_pos[2]

        height_progress = base_pos[2] - self._prev_height
        self._prev_height = base_pos[2]
        r_progress = 20 * max(0,height_progress)
        if base_pos[2] > 0.35:
            r_progress = 0.0
        if pose_dist < 0.15 and r_upright > 0.5:
            r_progress += 20 * max(0, height_progress) # Bonus for progress if already somewhat upright and posed

        #Term 7: join interlocking penalty:
        joint_penalty = -0.009 * np.sum(np.maximum(0, np.abs(cur_joints) - 2.5))

        # Term 8: Stability Bonus (Encourages staying in the zone)
        r_stability = 0.0
        if base_pos[2] > 0.28 and r_upright > 0.8:
            r_stability = 1.0
            
        # Term 9: Foot Contact (Encourages weight on feet, not knees)
        contact = sum(robot.GetFootContacts())
        r_feet = 0.09 * contact

        # Total Weighted Reward
        # 0.5 Pose + 0.3 Upright + 0.2 Height + 0.01 Energy - 0.1 Time
        total_reward = (0.5* r_pose) + (0.4 * r_upright) + (0.3 * r_height) + r_energy + r_time + joint_penalty + r_progress + self._standing_reward+ r_stability + r_feet
        
        if random.random() < 0.0001:
            print("Height:", base_pos[2], "Progress:", height_progress)

        return total_reward

    def done(self, env):
        """
        Terminates the episode if the robot leaves the 'chaos' and is 
        stable enough for the walking model to take over.
        """
        if not self._done_after_standing:
            return False

        # If upright, at correct height, and joints are settled, end episode (Success!)
        base_pos = env.robot.GetBasePosition()
        roll, pitch, _ = env.robot.GetTrueBaseRollPitchYaw()

        lin_vel = np.linalg.norm(env.robot.GetBaseVelocity())
        is_stable = lin_vel < 0.15
        
        is_upright = (abs(roll) < 0.1 and abs(pitch) < 0.1)
        is_high_enough = (0.28<base_pos[2] < 0.38)

        cur_joints = np.array(env.robot.GetMotorAngles())
        target_pose = np.array(self._target_pose)
        pose_dist = np.mean(np.square(cur_joints - target_pose))

        pose_good = pose_dist < 0.03

        success= is_upright and is_high_enough #and pose_good and is_stable
        if success and self._standing_reward == 0.0:
            self._standing_reward = 20.0
        self.last_success = success # Store for curriculum update
        
        if success:
            print(f"Episode Success! Height: {round(base_pos[2], 3)}, Roll: {round(roll, 3)}, Pitch: {round(pitch, 3)}, Pose Dist: {round(pose_dist, 4)}, Lin Vel: {round(lin_vel, 3)}")
            return True
        
        if env.env_step_counter > 600 and not success:
            print(f"Episode Failure. Height: {round(base_pos[2], 3)}, Roll: {round(roll, 3)}, Pitch: {round(pitch, 3)}, Pose Dist: {round(pose_dist, 4)}, Lin Vel: {round(lin_vel, 3)}")
            return True

        return success