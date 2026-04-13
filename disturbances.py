import numpy as np
import pybullet as p

class DisturbanceManager:
    def __init__(self, env):
        """
        env: The locomotion_gym_env instance.
        """
        self.env = env.unwrapped
        self.robot_id = self.env.robot.quadruped

    def apply_push(self, force_magnitude=500, direction='lateral', duration_steps=1):
        """
        Applies an external force to the robot base.
        direction: 'lateral' (Y-axis), 'frontal' (X-axis), or 'random'
        """
        if direction == 'lateral':
            force_vector = [0, force_magnitude, 0]
        elif direction == 'frontal':
            force_vector = [force_magnitude, 0, 0]
        else:
            # Random direction in the XY plane
            theta = np.random.uniform(0, 2 * np.pi)
            force_vector = [force_magnitude * np.cos(theta), force_magnitude * np.sin(theta), 0]

        # Apply the force to the base link (index -1)
        p.applyExternalForce(
            objectUniqueId=self.robot_id,
            linkIndex=-1,
            forceObj=force_vector,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME
        )
        print(f"\n[DISTURBANCE] Applied {force_magnitude}N {direction} push.")