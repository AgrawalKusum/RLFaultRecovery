import numpy as np
from motion_imitation.envs.env_wrappers import imitation_task
from pybullet_utils import transformations

from motion_imitation.envs.env_wrappers import imitation_terminal_conditions


class RecoveryImitationTask(imitation_task.ImitationTask):

    def __init__(self,
                 terminal_condition=imitation_terminal_conditions.imitation_terminal_condition,
                 disturbance_min=80,
                 disturbance_max=120,
                 disturbance_duration_min=0.1,
                 disturbance_duration_max=0.2,
                 disturbance_interval_min=0.8,
                 disturbance_interval_max=1.5,
                 recovery_bonus=2.0,
                 stability_bonus=0.5,
                 force_increment=15,
                 disturbance_cap=250,
                 mastery_threshold=0.8,
                 hysteresis=0.05,
                 mastery_window=10,
                 **kwargs):

        super().__init__( terminal_condition=terminal_condition, **kwargs)

        self._dist_min = disturbance_min
        self._dist_max = disturbance_max
        self._dur_min = disturbance_duration_min
        self._dur_max = disturbance_duration_max
        self._int_min = disturbance_interval_min
        self._int_max = disturbance_interval_max

        self._recovery_bonus = recovery_bonus
        self._stability_bonus = stability_bonus

        self._next_disturbance_time = 0
        self._disturbance_end_time = 0
        self._disturbance_force = None
        self._recovering = False
        self._force_active = False
        self._force_increment = force_increment
        self._dist_cap = disturbance_cap
        self._mastery_threshold = mastery_threshold
        self._hysteresis = hysteresis
        self._mastery_window = mastery_window
        self._success_count = 0
        self._disturbance_count = 0
        self._stable_counter = 0
        self._stable_steps_required=15
        
        #evaluation params
        self._recovery_times=[]
        self.max_torques=[]
        self.recovery_successes=0
        self.total_disturbances=0

        self._recovery_start_step=0
        self._episode_max_torque=0


    # ----------------------------------------------------

    def reset(self, env):
        if self._recovering:
            self._recovery_times.append(None)
            self.max_torques.append(self._episode_max_torque)

        super().reset(env)
        self._force_active = False
        self._recovering = False
        self._stable_counter = 0
        control_dt = env._sim_time_step * env._num_action_repeat
        self._stable_steps_required = int(0.5 / control_dt)
        self._schedule_next_disturbance(env)

    # ----------------------------------------------------

    def update(self, env):
        super().update(env)
        current_time = env.get_time_since_reset()

        # Phase 1: Disturbance is ACTIVELY being applied
        if self._next_disturbance_time <= current_time <= self._disturbance_end_time:
            self._apply_external_force(env)
            if not self._force_active:  # Trigger ONLY on the very first frame of the push
                self.total_disturbances += 1
                self._disturbance_count += 1
                self._recovery_start_step = env.get_step_counter()
                self._episode_max_torque = 0
                self._force_active = True
                self._recovering = True

        # Phase 2: Disturbance just ended, but robot is still recovering
        if self._force_active and current_time > self._disturbance_end_time:
            self._force_active = False
            # We DON'T reset recovering here; the reward function handles that
            self._schedule_next_disturbance(env)
            # Check mastery here if you want to include attempts that didn't stabilize yet
            # self._check_mastery() 

        # Ongoing Metric Tracking
        if self._recovering:
            torques = env.robot.GetMotorTorques()
            current_max = np.max(np.abs(torques))
            self._episode_max_torque = max(self._episode_max_torque, current_max)
    # ----------------------------------------------------

    def reward(self, env):

        imitation_reward = super().reward(env)

        stability_reward = self._compute_stability_bonus(env)

        recovery_reward = 0
        if self._recovering:
            if self._is_stable(env):
                self._stable_counter += 1
            else:
                self._stable_counter = 0

            if self._stable_counter >= self._stable_steps_required:

                recovery_reward = self._recovery_bonus

                recovery_time = env.get_step_counter() - self._recovery_start_step

                self._recovery_times.append(recovery_time)
                self.max_torques.append(self._episode_max_torque)
                self.recovery_successes += 1

                self._recovering = False
                self._stable_counter = 0
                self._success_count += 1
                self._check_mastery()

        total = imitation_reward + stability_reward + recovery_reward

        return total

    # ----------------------------------------------------

    def _schedule_next_disturbance(self, env):
        current_time = env.get_time_since_reset()

        interval = np.random.uniform(self._int_min, self._int_max)
        duration = np.random.uniform(self._dur_min, self._dur_max)

        magnitude = np.random.uniform(self._dist_min, self._dist_max)
        direction = np.random.uniform(-1, 1, size=3)
        direction[2] = 0  # horizontal push
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        else:
            direction = np.array([1,0,0])

        self._disturbance_force = magnitude * direction

        self._next_disturbance_time = current_time + interval
        self._disturbance_end_time = self._next_disturbance_time + duration

    # ----------------------------------------------------

    def _apply_external_force(self, env):
        pyb = env._pybullet_client
        robot_id = env.robot.quadruped

        base_pos, _ = pyb.getBasePositionAndOrientation(robot_id)

        pyb.applyExternalForce(
            objectUniqueId=robot_id,
            linkIndex=-1,
            forceObj=self._disturbance_force,
            posObj= [0,0,0],
            flags=pyb.WORLD_FRAME
        )

        if hasattr(env, '_is_render') and env._is_render:
            magnitude = np.linalg.norm(self._disturbance_force)
            force_visual_scale = 0.005 
            
            line_end = [
                base_pos[0] + self._disturbance_force[0] * force_visual_scale,
                base_pos[1] + self._disturbance_force[1] * force_visual_scale,
                base_pos[2] + self._disturbance_force[2] * force_visual_scale
            ]
            print(f"DEBUG: Pushing with {self._disturbance_force} at time {env.get_time_since_reset()}")
            # Draw the Red Arrow
            pyb.addUserDebugLine(
                lineFromXYZ=base_pos,
                lineToXYZ=line_end,
                lineColorRGB=[1, 0, 0],
                lineWidth=4,
                lifeTime=0.05
            )

            # Draw the Magnitude Text (Yellow)
            # Offset the text slightly upward ([0, 0, 0.2]) so it doesn't overlap the robot
            text_pos = [base_pos[0], base_pos[1], base_pos[2] + 0.2]
            pyb.addUserDebugText(
                text=f"{magnitude:.1f} N",
                textPosition=text_pos,
                textColorRGB=[1, 0, 0], # Yellow
                textSize=1.2,
                lifeTime=0.05
            )
    # ----------------------------------------------------

    def _compute_stability_bonus(self, env):
        pyb = env._pybullet_client
        robot_id = env.robot.quadruped

        _, base_rot = pyb.getBasePositionAndOrientation(robot_id)
        roll, pitch, _ = transformations.euler_from_quaternion(base_rot)

        tilt = abs(roll) + abs(pitch)

        return self._stability_bonus * np.exp(-5 * tilt)

    # ----------------------------------------------------

    def _is_stable(self, env):
        pyb = env._pybullet_client
        robot_id = env.robot.quadruped

        _, base_rot = pyb.getBasePositionAndOrientation(robot_id)
        roll, pitch, _ = transformations.euler_from_quaternion(base_rot)

        return abs(roll) < 0.15 and abs(pitch) < 0.15
    
    def _check_mastery(self):

        if self._disturbance_count < self._mastery_window:
            return

        success_rate = self._success_count / self._disturbance_count

        if success_rate > self._mastery_threshold:
            self._dist_max = min(self._dist_max + self._force_increment,
                                 self._dist_cap)

        elif success_rate < (self._mastery_threshold - self._hysteresis):
            self._dist_max = max(self._dist_min,
                                 self._dist_max - self._force_increment)

        # Reset window
        self._success_count = 0
        self._disturbance_count = 0

    def get_metrics(self):
        success_rate = (self.recovery_successes / self.total_disturbances) if self.total_disturbances > 0 else 0
        
        # Filter out None values from recovery times (None usually means it fell)
        valid_times = [t for t in self._recovery_times if t is not None]
        avg_recovery_time = np.mean(valid_times) if valid_times else 0
        
        return {
            "total_disturbances": self.total_disturbances,
            "success_rate": success_rate,
            "avg_recovery_time_steps": avg_recovery_time,
            "max_force_limit": self._dist_max,
            "peak_torques": self.max_torques
        }