import time

import numpy as np
import numpy.typing as npt
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.utils import make_homogenous_matrix_from_rotation_matrix


def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> None:
    if arm == "r_arm":
        ik = reachy.r_arm.inverse_kinematics(pose)
        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
    elif arm == "l_arm":
        ik = reachy.l_arm.inverse_kinematics(pose)
        for joint, goal_pos in zip(reachy.l_arm.joints.values(), ik):
            joint.goal_position = goal_pos


def main_test() -> None:
    print("Trying to connect on localhost Reachy...")
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    # Go to a specific pose with the right arm
    goal_position = [0.55, -0.3, -0.15]
    goal_orientation = [0, -np.pi / 2, 0]
    rotation_matrix = R.from_euler("xyz", goal_orientation).as_matrix()
    goal_pose = make_homogenous_matrix_from_rotation_matrix(goal_position, rotation_matrix)
    go_to_pose(reachy, goal_pose, "r_arm")

    time.sleep(1.0)

    # Go to a specific pose with the left arm
    goal_position = [0.55, 0.3, -0.15]
    goal_orientation = [0, -np.pi / 2, 0]
    rotation_matrix = R.from_euler("xyz", goal_orientation).as_matrix()
    goal_pose = make_homogenous_matrix_from_rotation_matrix(goal_position, rotation_matrix)
    go_to_pose(reachy, goal_pose, "l_arm")

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


if __name__ == "__main__":
    main_test()
