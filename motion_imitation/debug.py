# Simple script to trigger 'done' and 'update_curriculum' immediately
from motion_imitation.envs.env_wrappers.sac_recovery_task import SACRecoveryTask

class MockEnv:
    def __init__(self):
        self.robot = type('obj', (object,), {
            'GetBasePosition': lambda: [0, 0, 0.45], # Force success height
            'GetTrueBaseRollPitchYaw': lambda: [0, 0, 0], # Force upright
            'GetMotorAngles': lambda: [0]*12 # Force pose
        })
        self.unwrapped = self
        self._task = None # Will be set below

task = SACRecoveryTask(target_pose=[0]*12)
env = MockEnv()
env._task = task

# TEST 1: Check if done() works without AttributeError
print("Testing done()...")
is_done = task.done(env)
print(f"Success: {is_done}, Task internal flag: {task.last_success}")

# TEST 2: Check curriculum update
print("Testing update_curriculum()...")
task.update_curriculum(success=True, steps_at_level=100)
print(f"Curriculum Level: {task._curriculum_level}")

print("All local tests passed!")