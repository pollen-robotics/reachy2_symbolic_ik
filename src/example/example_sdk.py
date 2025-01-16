import time

import numpy as np
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R
from reachy2_symbolic_ik.utils import make_homogenous_matrix_from_rotation_matrix

def sdk_basics() -> None:
    # Create the ReachySDK object
    print("Trying to connect on localhost Reachy...")
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    # Define the goal pose
    goal_position = [0.55, -0.3, -0.15]
    goal_orientation = [0, -np.pi / 2, 0]
    goal_pose = np.array([goal_position, goal_orientation])
    goal_pose = make_homogenous_matrix_from_rotation_matrix(goal_position, R.from_euler("xyz", goal_orientation).as_matrix())

    # Get joints for the goal pose
    joints = reachy.r_arm.inverse_kinematics(goal_pose)

    # Go to the goal pose
    reachy.r_arm.goto(joints, duration=4.0, degrees=True, interpolation_mode="minimum_jerk", wait=True)


if __name__ == "__main__":
    sdk_basics()


