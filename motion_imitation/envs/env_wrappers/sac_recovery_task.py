import numpy as np
import random

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

    def custom_reset(self, env):
        """
        The 'Chaos' Reset: Randomizes 3D Orientation and Joint Angles.
        This is called by your modified LocomotionGymEnv.reset().
        """
        # 1. Randomize Orientation (Full 3D Chaos)
        angle_limit = self._get_chaos_limits()
        random_roll = random.uniform(-angle_limit, angle_limit)
        random_pitch = random.uniform(-angle_limit, angle_limit)
        random_yaw = random.uniform(-np.pi, np.pi)
        
        chaos_quat = env.pybullet_client.getQuaternionFromEuler([
            random_roll, random_pitch, random_yaw
        ])

        # 2. Randomize Position (Slightly elevated to prevent clipping)
        random_pos = [0, 0, 0.5] 

        # 3. Randomize Joints (Leg Entanglement)
        # We pick angles within the robot's physical limits
        joint_limit = 0.2 + (self._curriculum_level * 0.6)
        random_joints = [random.uniform(-joint_limit, joint_limit) for _ in range(12)]

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

    def reward(self, env):
        """
        Multi-term reward for SAC to learn recovery.
        """
        robot = env.robot
        # Get current state
        cur_joints = robot.GetMotorAngles()
        roll, pitch, _ = robot.GetTrueBaseRollPitchYaw()
        up_vector = robot.GetTrueBaseOrientation() # Quaternion
        
        # Term 1: Pose Reward (How close are we to the 'Ready' stance?)
        # Use negative MSE between current and target joints
        pose_dist = np.mean(np.square(cur_joints - self._target_pose))
        r_pose = np.exp(-5.0 * pose_dist)

        # Term 2: Uprightness Reward (Penalty for being tilted)
        # We want roll and pitch to be near 0
        orientation_dist = np.sqrt(roll**2 + pitch**2)
        r_upright = np.exp(-2.0 * orientation_dist)

        # Term 3: Height Reward
        # Encourage the base to be at a standing height (~0.45m for Laikago)
        #base_pos, _ = robot.GetBasePositionAndOrientation()
        base_pos = robot.GetBasePosition()
        base_orn = robot.GetBaseOrientation()
        r_height = np.exp(-10.0 * (base_pos[2] - 0.45)**2)

        # Term 4: Energy Penalty
        # Discourage spastic leg movements (helps SAC converge on smooth motions)
        r_energy =  np.exp(-0.1 * np.mean(np.square(robot.GetMotorVelocities())))

        #Term 5: Time Penalty
        # Encourage faster recovery by penalizing time taken
        r_time = -0.1
        # Total Weighted Reward
        # 0.5 Pose + 0.3 Upright + 0.2 Height + 0.01 Energy - 0.1 Time
        total_reward = (0.5 * r_pose) + (0.3 * r_upright) + (0.2 * r_height) + (0.01 * r_energy) + r_time
        
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
        base_orn = env.robot.GetBaseOrientation()
        roll, pitch, _ = env.robot.GetTrueBaseRollPitchYaw()
        
        is_upright = (abs(roll) < 0.1 and abs(pitch) < 0.1)
        is_high_enough = (base_pos[2] > 0.35)

        cur_joints = env.robot.GetMotorAngles()
        pose_dist = np.mean(np.square(cur_joints - self._target_pose))

        pose_good = pose_dist < 0.05
        
        if is_upright and is_high_enough and pose_good:
            # This info dict is what the Callback reads!
            env._info["is_success"] = True 
            return True
        return False