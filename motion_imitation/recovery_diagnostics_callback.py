from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RecoveryDiagnosticsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:

        # Access first env (works for VecEnv)
        env = self.training_env.envs[0]
        robot = env.robot

        # Base height
        base_pos = robot.GetBasePosition()
        height = base_pos[2]

        # Orientation
        roll, pitch, yaw = robot.GetTrueBaseRollPitchYaw()

        # Pose error
        joints = np.array(robot.GetMotorAngles())
        target = np.array(env.task._target_pose)

        pose_error = np.mean((joints - target) ** 2)

        # Log to TensorBoard
        self.logger.record("recovery/base_height", height)
        self.logger.record("recovery/roll", roll)
        self.logger.record("recovery/pitch", pitch)
        self.logger.record("recovery/pose_error", pose_error)

        return True