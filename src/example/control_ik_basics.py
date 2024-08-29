import time

import numpy as np
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.control_ik import ControlIK
from reachy2_symbolic_ik.utils import make_homogenous_matrix_from_rotation_matrix


def discrete_test() -> None:
    pass


def continuous_test() -> None:
    pass


def test_go_to(ik: ControlIK, reachy: ReachySDK) -> None:
    r_goal_poses = np.array(
        [
            [[0.0, -0.2, -0.66], [0, 0, 0]],
            [[0.0, -0.86, -0.0], [-np.pi / 2, 0, 0]],
            [[0.0, -0.58, -0.28], [-np.pi / 2, -np.pi / 2, 0]],
            [[0.38, -0.2, -0.28], [0, -np.pi / 2, 0]],
            [[0.66, -0.2, -0.0], [0, -np.pi / 2, 0]],
            [[0.0, -0.2, -0.66], [0, 0, 0]],
            # [[0.2, 0.20, -0.18], [np.pi / 2, -np.pi / 2, 0]],
            # [[0.10, 0.20, -0.22], [np.pi / 3, -np.pi / 2, 0]],
            # [[0.10, 0.25, -0.22], [np.pi / 3, -np.pi / 2, 0]],
        ]
    )
    control_types = ["discrete", "continuous"]

    for control_type in control_types:
        print(f"Control type: {control_type}")

        for r_goal_pose in r_goal_poses:
            rotation_matrix = R.from_euler("xyz", r_goal_pose[1]).as_matrix()
            goal_pose = make_homogenous_matrix_from_rotation_matrix(r_goal_pose[0], rotation_matrix)

            joints, is_reachable, state = ik.symbolic_inverse_kinematics("r_arm", goal_pose, control_type)

            joints = np.degrees(joints)

            print(f"joints {joints}")
            print(f"is_reachable {is_reachable}")

            for joint, goal_pos in zip(reachy.r_arm.joints.values(), joints):
                joint.goal_position = goal_pos

            # for joint, goal_pos in zip(reachy.r_arm.joints.values(), np.degrees(joints)):
            #     joint.goal_position = goal_pos

            time.sleep(2.0)


# def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> None:
#     if arm == "r_arm":
#         ik = reachy.r_arm.inverse_kinematics(pose)
#         real_pose = reachy.r_arm.forward_kinematics(ik)
#         # print(f"pose by kdl {real_pose}")
#         pose_diff = np.linalg.norm(pose - real_pose)
#         print(f"pose diff {pose_diff}")
#         if pose_diff > 0.001:
#             print(f"pose by kdl {real_pose}")
#         for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
#             joint.goal_position = goal_pos
#     elif arm == "l_arm":
#         ik = reachy.l_arm.inverse_kinematics(pose)
#         real_pose = reachy.l_arm.forward_kinematics(ik)
#         # print(f"pose by kdl {poreal_posese}")
#         pose_diff = np.linalg.norm(pose - real_pose)
#         print(f"pose diff {pose_diff}")
#         if pose_diff > 0.005:
#             print(f"pose by kdl {real_pose}")
#         for joint, goal_pos in zip(reachy.l_arm.joints.values(), ik):
#             joint.goal_position = goal_pos


def main_test() -> None:
    ik_control = ControlIK(urdf_path="../config_files/reachy2.urdf")
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    # ik_control.test()
    test_go_to(ik_control, reachy)


if __name__ == "__main__":
    main_test()
