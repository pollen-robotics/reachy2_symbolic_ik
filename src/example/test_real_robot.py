import time

import numpy as np
import numpy.typing as npt
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.utils import make_homogenous_matrix_from_rotation_matrix


def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> None:
    pose = make_homogenous_matrix_from_rotation_matrix(pose[0], R.from_euler("xyz", pose[1]).as_matrix())
    if arm == "r_arm":
        ik = reachy.r_arm.inverse_kinematics(pose)
        reachy.r_arm.goto_joints(ik, 4.0, degrees=True, interpolation_mode="minimum_jerk")
    elif arm == "l_arm":
        ik = reachy.l_arm.inverse_kinematics(pose)
        reachy.l_arm.goto_joints(ik, 4.0, degrees=True, interpolation_mode="minimum_jerk")


def test_movement(reachy: ReachySDK) -> None:
    go_to_pose(reachy, np.array([[0.0, 0.2, -0.66], [0.0, 0.0, 0.0]]), "l_arm")
    go_to_pose(reachy, np.array([[0.0, -0.2, -0.66], [0.0, 0.0, 0.0]]), "r_arm")
    time.sleep(5)
    go_to_pose(reachy, np.array([[0.38, -0.2, -0.28], [0.0, -np.pi / 2, 0.0]]), "r_arm")
    time.sleep(5)
    go_to_pose(reachy, np.array([[0.20, -0.025, -0.28], [0.0, -np.pi / 2, np.pi / 4]]), "r_arm")
    time.sleep(5)
    go_to_pose(reachy, np.array([[0.38, -0.2, -0.28], [0.0, -np.pi / 2, 0.0]]), "r_arm")
    time.sleep(5)
    go_to_pose(reachy, np.array([[0.68, -0.2, -0.0], [0.0, -np.pi / 2, 0.0]]), "r_arm")
    time.sleep(5)
    go_to_pose(reachy, np.array([[0.0, -0.2, -0.66], [0.0, 0.0, 0.0]]), "r_arm")
    time.sleep(5)
    go_to_pose(reachy, np.array([[0.0, 0.2, -0.66], [0.0, 0.0, 0.0]]), "l_arm")
    time.sleep(5)
    go_to_pose(reachy, np.array([[0.38, 0.2, -0.28], [0.0, -np.pi / 2, 0.0]]), "l_arm")
    time.sleep(5)
    go_to_pose(reachy, np.array([[0.20, -0.025, -0.28], [0.0, -np.pi / 2, -np.pi / 4]]), "l_arm")
    time.sleep(5)
    go_to_pose(reachy, np.array([[0.38, 0.2, -0.28], [0.0, -np.pi / 2, 0.0]]), "l_arm")
    time.sleep(5)
    go_to_pose(reachy, np.array([[0.68, 0.2, -0.0], [0.0, -np.pi / 2, 0.0]]), "l_arm")
    time.sleep(5)
    go_to_pose(reachy, np.array([[0.0, 0.2, -0.66], [0.0, 0.0, 0.0]]), "l_arm")
    time.sleep(5)


def go_to_zero(reachy: ReachySDK) -> None:
    go_to_pose(reachy, np.array([[0.0, 0.2, -0.66], [0.0, 0.0, 0.0]]), "l_arm")
    go_to_pose(reachy, np.array([[0.0, -0.2, -0.66], [0.0, 0.0, 0.0]]), "r_arm")


def main() -> None:
    reachy = ReachySDK(host="localhost")

    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    go_to_zero(reachy)

    go_to_pose(reachy, np.array([[0.38, -0.2, -0.28], [0.0, -np.pi / 2, 0.0]]), "r_arm")
    time.sleep(10)

    # test_movement(reachy)

    reachy.disconnect()


if __name__ == "__main__":
    main()
