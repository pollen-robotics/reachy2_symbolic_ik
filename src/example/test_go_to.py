import time

import numpy as np
import numpy.typing as npt
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import (
    is_elbow_ok,
    make_homogenous_matrix_from_rotation_matrix,
)


def go_to_joint_positions(reachy: ReachySDK, joint_positions: npt.NDArray[np.float64], arm: str) -> None:
    if arm == "r_arm":
        for joint, goal_pos in zip(reachy.r_arm.joints.values(), joint_positions):
            joint.goal_position = goal_pos
    elif arm == "l_arm":
        for joint, goal_pos in zip(reachy.l_arm.joints.values(), joint_positions):
            joint.goal_position = goal_pos


def go_to_pose_with_all_theta(
    reachy: ReachySDK, symbolic_ik: SymbolicIK, pose: npt.NDArray[np.float64], arm: str, nbr_points: int = 50
) -> None:
    is_reachable, interval, get_joints, _ = symbolic_ik.is_reachable(pose)
    print(f"interval {interval}")
    if interval[0] > interval[1]:
        interval = [interval[0], interval[1] + 2 * np.pi]
    print(interval)
    if is_reachable:
        for i in range(nbr_points):
            theta = interval[0] + i * (interval[1] - interval[0]) / nbr_points
            joints, elbow_position = get_joints(theta)
            # print(f"joints {joints}")
            if arm == "r_arm":
                if is_elbow_ok(elbow_position, 1):
                    print("Elbow is ok")
                else:
                    print("Elbow is not ok")
                real_pose = reachy.r_arm.forward_kinematics(np.degrees(joints))
                goal_pose_matrix = make_homogenous_matrix_from_rotation_matrix(
                    pose[0], R.from_euler("xyz", pose[1]).as_matrix()
                )
                # print(f"pose by kdl {real_pose}")
                goal_diff = np.linalg.norm(goal_pose_matrix - real_pose)
                print(f"goal diff {goal_diff}")
                diff = np.linalg.norm(pose[0] - real_pose[:3, 3])
                print(f"goal diff xyz only {diff}m")
                for joint, goal_pos in zip(reachy.r_arm.joints.values(), np.degrees(joints)):
                    joint.goal_position = goal_pos
                time.sleep(0.1)
            elif arm == "l_arm":
                if is_elbow_ok(elbow_position, -1):
                    print("Elbow is ok")
                else:
                    print("Elbow is not ok")
                real_pose = reachy.l_arm.forward_kinematics(np.degrees(joints))
                goal_pose_matrix = make_homogenous_matrix_from_rotation_matrix(
                    pose[0], R.from_euler("xyz", pose[1]).as_matrix()
                )
                # print(f"pose by kdl {real_pose}")
                goal_diff = np.linalg.norm(goal_pose_matrix - real_pose)
                # print(f"goal diff {goal_diff}")
                # goal diff xyz only
                diff = np.linalg.norm(pose[0] - real_pose[:3, 3])
                print(f"goal diff xyz only {diff}m")
                for joint, goal_pos in zip(reachy.l_arm.joints.values(), np.degrees(joints)):
                    joint.goal_position = goal_pos
                time.sleep(0.1)
    else:
        print("Pose not reachable")


def go_to_pose_with_choosen_theta(
    reachy: ReachySDK, symbolic_ik: SymbolicIK, pose: npt.NDArray[np.float64], theta: float, arm: str
) -> None:
    is_reachable, interval, get_joints = symbolic_ik.is_reachable(pose)
    if is_reachable:
        joints, elbow_position = get_joints(theta)
        print(f"joints {joints}")
        if arm == "r_arm":
            real_pose = reachy.r_arm.forward_kinematics(np.degrees(joints))
            goal_pose_matrix = make_homogenous_matrix_from_rotation_matrix(pose[0], R.from_euler("xyz", pose[1]).as_matrix())
            # print(f"pose by kdl {real_pose}")
            goal_diff = np.linalg.norm(goal_pose_matrix - real_pose)
            print(f"goal diff {goal_diff}")
            for joint, goal_pos in zip(reachy.r_arm.joints.values(), np.degrees(joints)):
                joint.goal_position = goal_pos
            time.sleep(1)
        elif arm == "l_arm":
            real_pose = reachy.r_arm.forward_kinematics(np.degrees(joints))
            goal_pose_matrix = make_homogenous_matrix_from_rotation_matrix(pose[0], R.from_euler("xyz", pose[1]).as_matrix())
            # print(f"pose by kdl {real_pose}")
            goal_diff = np.linalg.norm(goal_pose_matrix - real_pose)
            print(f"goal diff {goal_diff}")
            for joint, goal_pos in zip(reachy.l_arm.joints.values(), np.degrees(joints)):
                joint.goal_position = goal_pos
            time.sleep(1)
    else:
        print("Pose not reachable")


def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> None:
    if arm == "r_arm":
        ik = reachy.r_arm.inverse_kinematics(pose)
        real_pose = reachy.r_arm.forward_kinematics(ik)
        # print(f"pose by kdl {real_pose}")
        pose_diff = np.linalg.norm(pose - real_pose)
        print(f"pose diff {pose_diff}")
        if pose_diff > 0.001:
            print(f"pose by kdl {real_pose}")
        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
    elif arm == "l_arm":
        ik = reachy.l_arm.inverse_kinematics(pose)
        real_pose = reachy.l_arm.forward_kinematics(ik)
        # print(f"pose by kdl {poreal_posese}")
        pose_diff = np.linalg.norm(pose - real_pose)
        print(f"pose diff {pose_diff}")
        if pose_diff > 0.005:
            print(f"pose by kdl {real_pose}")
        for joint, goal_pos in zip(reachy.l_arm.joints.values(), ik):
            joint.goal_position = goal_pos


def test_poses(reachy: ReachySDK, r_symbolic_ik: SymbolicIK, l_symbolic_ik: SymbolicIK) -> None:
    r_goal_poses = np.array(
        [
            [[0.0, -0.2, -0.66], [0, 0, 0]],
            [[0.0, -0.86, -0.0], [-np.pi / 2, 0, 0]],
            [[0.0, -0.58, -0.28], [-np.pi / 2, -np.pi / 2, 0]],
            [[0.38, -0.2, -0.28], [0, -np.pi / 2, 0]],
            [[0.66, -0.2, -0.0], [0, -np.pi / 2, 0]],
            [[0.0, -0.2, -0.66], [0, 0, 0]],
        ]
    )
    l_goal_poses = np.array(
        [
            [[0.0, 0.2, -0.66], [0, 0, 0]],
            [[0.0, 0.86, -0.0], [np.pi / 2, 0, 0]],
            [[0.0, 0.58, -0.28], [np.pi / 2, -np.pi / 2, 0]],
            [[0.38, 0.2, -0.28], [0, -np.pi / 2, 0]],
            [[0.66, 0.2, -0.0], [0, -np.pi / 2, 0]],
            [[0.0, 0.2, -0.66], [0, 0, 0]],
        ]
    )

    for r_goal_pose in r_goal_poses:
        is_reachable, interval, get_joints = r_symbolic_ik.is_reachable(r_goal_pose)
        print(f"Is reachable {is_reachable}")
        rotation_matrix = R.from_euler("xyz", r_goal_pose[1]).as_matrix()
        goal_pose = make_homogenous_matrix_from_rotation_matrix(r_goal_pose[0], rotation_matrix)
        go_to_pose(reachy, goal_pose, "r_arm")
        time.sleep(2.0)

    for l_goal_pose in l_goal_poses:
        is_reachable, interval, get_joints = l_symbolic_ik.is_reachable(l_goal_pose)
        print(f"Is reachable {is_reachable}")
        rotation_matrix = R.from_euler("xyz", l_goal_pose[1]).as_matrix()
        goal_pose = make_homogenous_matrix_from_rotation_matrix(l_goal_pose[0], rotation_matrix)
        go_to_pose(reachy, goal_pose, "l_arm")
        time.sleep(2.0)


def null_space_test() -> None:
    print("Trying to connect on localhost Reachy...")
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    # symbolic_ik_r = SymbolicIK(arm="r_arm", upper_arm_size=0.28, forearm_size=0.28, gripper_size=0.10, wrist_limit=42.5)
    symbolic_ik_l = SymbolicIK(arm="l_arm", upper_arm_size=0.28, forearm_size=0.28, gripper_size=0.10, wrist_limit=42.5)

    goal_pose = np.array(
        [
            [0.36861, 0.089736, -0.92524, 0.37213],
            [-0.068392, 0.99525, 0.069279, -0.028012],
            [0.92706, 0.037742, 0.373, -0.38572],
            [0, 0, 0, 1],
        ]
    )
    # transform to  goal_position, goal_orientation
    goal_position = goal_pose[:3, 3]
    goal_orientation = R.from_matrix(goal_pose[:3, :3]).as_euler("xyz")
    pose = [goal_position, goal_orientation]
    go_to_pose_with_all_theta(reachy, symbolic_ik_l, pose, "l_arm")
    time.sleep(1.0)

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


def main_test() -> None:
    print("Trying to connect on localhost Reachy...")
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    symbolic_ik_r = SymbolicIK(shoulder_orientation_offset=[10, 0, 15], elbow_orientation_offset=[0, 0, 0])
    symbolic_ik_l = SymbolicIK(arm="l_arm", shoulder_orientation_offset=[10, 0, 15], elbow_orientation_offset=[0, 0, 0])

    test_poses(reachy, symbolic_ik_r, symbolic_ik_l)

    # go_to_joint_positions(reachy, [0, 0, 0, 0, 0, 0, 0], "r_arm")
    # go_to_joint_positions(reachy, [0, 0, 0, 0, 0, 0, 0], "l_arm")
    # time.sleep(1.0)
    # while True:
    #     go_to_joint_positions(reachy, [0, 0, 0, 0, 20, 0, 0], "r_arm")
    #     go_to_joint_positions(reachy, [0, 0, 0, 0, 20, 0, 0], "l_arm")
    #     print("+20")
    #     time.sleep(1.0)
    #     go_to_joint_positions(reachy, [0, 0, 0, 0, -20, 0, 0], "r_arm")
    #     go_to_joint_positions(reachy, [0, 0, 0, 0, -20, 0, 0], "l_arm")
    #     print("-20")
    #     time.sleep(1.0)

    time.sleep(1.0)

    # Rigth arm
    # Go to a specific pose with the right arm
    # goal_position = [0.0, -0.86, -0.0]
    # goal_position = [0.0, -0.58, -0.28]
    # goal_orientation = [-np.pi / 2, 0, 0]
    # goal_position = [0.58, -0.2, -0.0]
    # goal_position = [0.37, -0.2, -0.28]

    # goal_position = [0.0, -0.2, -0.68]
    # goal_orientation = [0, -np.pi / 2, 0]
    # goal_position = [0.0, -0.86, -0.0]
    goal_position = [0.0, -0.8, -0.24]
    goal_orientation = [-np.pi / 2, -np.pi / 2, 0]
    # goal_position = [0.0, -0.2, -0.66]
    # goal_orientation = [0, 0, np.radians(120)]

    goal_pose = np.array([goal_position, goal_orientation])
    theta = -4 * np.pi / 5
    go_to_pose_with_choosen_theta(reachy, symbolic_ik_r, goal_pose, theta, "r_arm")
    time.sleep(2.0)
    go_to_pose_with_all_theta(reachy, symbolic_ik_r, goal_pose, "r_arm")
    time.sleep(1.0)

    # Left arm

    # goal_position = [0.0, 0.86, -0.0]
    goal_position = [0.0, 0.55, -0.28]
    goal_orientation = [np.pi / 2, 0, 0]
    # goal_position = [0.0, -0.2, -0.66]
    # goal_orientation = [0, 0, np.radians(120)]
    goal_pose = np.array([goal_position, goal_orientation])
    theta = -4 * np.pi / 5
    go_to_pose_with_choosen_theta(reachy, symbolic_ik_l, goal_pose, theta, "l_arm")
    time.sleep(2.0)
    go_to_pose_with_all_theta(reachy, symbolic_ik_l, goal_pose, "l_arm")
    time.sleep(1.0)

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


if __name__ == "__main__":
    main_test()
    # null_space_test()
