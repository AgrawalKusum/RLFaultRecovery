import numpy as np
from motion_imitation.envs.env_wrappers import imitation_task
from pybullet_utils import transformations

from motion_imitation.motion_imitation.envs.env_wrappers import imitation_terminal_conditions


class RecoveryImitationTask(imitation_task.ImitationTask):

    def __init__(self,
                 terminal_condition=imitation_terminal_conditions.imitation_terminal_condition,
                 disturbance_min=50,
                 disturbance_max=100,
                 disturbance_duration_min=0.02,
                 disturbance_duration_max=0.08,
                 disturbance_interval_min=1.0,
                 disturbance_interval_max=3.0,
                 recovery_bonus=2.0,
                 stability_bonus=0.5,
                 force_increment=10,
                 disturbance_cap=250,
                 mastery_threshold=0.8,
                 hysteresis=0.05,
                 mastery_window=20,
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
        self.force_active = False
        self._force_increment = force_increment
        self._dist_cap = disturbance_cap
        self._mastery_threshold = mastery_threshold
        self._hysteresis = hysteresis
        self._mastery_window = mastery_window
        self._success_count = 0
        self._disturbance_count = 0

    # ----------------------------------------------------

    def reset(self, env):
        super().reset(env)
        self._force_active = False
        self._recovering = False
        self._schedule_next_disturbance()

    # ----------------------------------------------------

    def update(self, env):
        super().update(env)

        current_time = env.get_time_since_reset()

        # Apply disturbance
        if self._next_disturbance_time <= current_time <= self._disturbance_end_time:
            self._apply_external_force(env)
            self._force_active = True
            self._recovering = True
        if self._force_active and current_time > self._disturbance_end_time:
            self._force_active = False
            self._disturbance_count += 1
            self._schedule_next_disturbance(env)
    # ----------------------------------------------------

    def reward(self, env):

        imitation_reward = super().reward(env)

        stability_reward = self._compute_stability_bonus(env)

        recovery_reward = 0
        if self._recovering and self._is_stable(env):
            recovery_reward = self._recovery_bonus
            self._recovering = False
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
            posObj=base_pos,
            flags=pyb.WORLD_FRAME
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