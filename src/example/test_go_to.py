import time

import numpy as np
import numpy.typing as npt
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.control_ik import ControlIK
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
    reachy.send_goal_positions()


def go_to_pose_with_all_theta(
    reachy: ReachySDK, symbolic_ik: SymbolicIK, pose: npt.NDArray[np.float64], arm: str, nbr_points: int = 50
) -> None:
    is_reachable, interval, get_joints, _ = symbolic_ik.is_reachable(pose)
    print(f"interval {interval}")
    if interval[0] > interval[1]:
        interval = [interval[0], interval[1] + 2 * np.pi]
    # print(interval)
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
                # real_pose = reachy.r_arm.forward_kinematics(np.degrees(joints))
                # goal_pose_matrix = make_homogenous_matrix_from_rotation_matrix(
                #     pose[0], R.from_euler("xyz", pose[1]).as_matrix()
                # )
                # print(f"pose by kdl {real_pose}")
                # goal_diff = np.linalg.norm(goal_pose_matrix - real_pose)
                # print(f"goal diff {goal_diff}")
                # diff = np.linalg.norm(pose[0] - real_pose[:3, 3])
                # print(f"goal diff xyz only {diff}m")
                for joint, goal_pos in zip(reachy.r_arm.joints.values(), np.degrees(joints)):
                    joint.goal_position = goal_pos
                reachy.send_goal_positions()
                time.sleep(0.1)
            elif arm == "l_arm":
                if is_elbow_ok(elbow_position, -1):
                    print("Elbow is ok")
                else:
                    print("Elbow is not ok")
                # real_pose = reachy.l_arm.forward_kinematics(np.degrees(joints))
                # goal_pose_matrix = make_homogenous_matrix_from_rotation_matrix(
                #     pose[0], R.from_euler("xyz", pose[1]).as_matrix()
                # )
                # print(f"pose by kdl {real_pose}")
                # goal_diff = np.linalg.norm(goal_pose_matrix - real_pose)
                # print(f"goal diff {goal_diff}")
                # goal diff xyz only
                # diff = np.linalg.norm(pose[0] - real_pose[:3, 3])
                # print(f"goal diff xyz only {diff}m")
                for joint, goal_pos in zip(reachy.l_arm.joints.values(), np.degrees(joints)):
                    joint.goal_position = goal_pos
                reachy.send_goal_positions()
                time.sleep(0.1)
    else:
        print("Pose not reachable")


def go_to_pose_with_choosen_theta(
    reachy: ReachySDK, symbolic_ik: SymbolicIK, pose: npt.NDArray[np.float64], theta: float, arm: str
) -> None:
    is_reachable, interval, get_joints, _ = symbolic_ik.is_reachable(pose)
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
            reachy.send_goal_positions()
            time.sleep(1)
        elif arm == "l_arm":
            real_pose = reachy.r_arm.forward_kinematics(np.degrees(joints))
            goal_pose_matrix = make_homogenous_matrix_from_rotation_matrix(pose[0], R.from_euler("xyz", pose[1]).as_matrix())
            # print(f"pose by kdl {real_pose}")
            goal_diff = np.linalg.norm(goal_pose_matrix - real_pose)
            print(f"goal diff {goal_diff}")
            for joint, goal_pos in zip(reachy.l_arm.joints.values(), np.degrees(joints)):
                joint.goal_position = goal_pos
            reachy.send_goal_positions()
            time.sleep(1)
    else:
        print("Pose not reachable")


def go_to_pose(
    reachy: ReachySDK,
    pose: npt.NDArray[np.float64],
    arm: str,
    controle_type: str = "discrete",
    constrained_mode: str = "unconstrained",
    verbose: bool = True,
) -> None:
    controle_ik = ControlIK()
    if arm == "r_arm":
        ik, is_reachable, state = controle_ik.symbolic_inverse_kinematics(
            "r_arm", pose, controle_type, constrained_mode=constrained_mode
        )
        ik = np.degrees(ik)
        real_pose = reachy.r_arm.forward_kinematics(ik)
        pose_diff = np.linalg.norm(pose - real_pose)
        if verbose:
            if is_reachable:
                print("\033[92m" + "Pose reachable" + "\033[0m")
            else:
                print("\033[91m" + "Pose not reachable" + "\033[0m")
            print(f"State {state}")
            print(f"pose diff {pose_diff}")
            # if pose_diff > 0.0015:
            #     print(f"fk {real_pose}")
        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        reachy.send_goal_positions()

    elif arm == "l_arm":
        ik, is_reachable, state = controle_ik.symbolic_inverse_kinematics(
            "l_arm", pose, controle_type, constrained_mode=constrained_mode
        )
        ik = np.degrees(ik)
        real_pose = reachy.l_arm.forward_kinematics(ik)
        pose_diff = np.linalg.norm(pose - real_pose)
        if verbose:
            if is_reachable:
                print("\033[92m" + "Pose reachable" + "\033[0m")
            else:
                print("\033[91m" + "Pose not reachable" + "\033[0m")
            print(f"State {state}")
            print(f"pose diff {pose_diff}")
            # if pose_diff > 0.005:
            #     print(f"pose by kdl {real_pose}")
        for joint, goal_pos in zip(reachy.l_arm.joints.values(), ik):
            joint.goal_position = goal_pos
        reachy.send_goal_positions()
    print("")


def test_poses(
    reachy: ReachySDK,
    r_symbolic_ik: SymbolicIK,
    l_symbolic_ik: SymbolicIK,
    controle_type: str = "discrete",
    constrained_mode: str = "unconstrained",
) -> None:
    r_goal_poses = np.array(
        [
            # reachable poses
            [[0.0001, -0.2, -0.6599], [0, 0, 0]],
            [[0.38, -0.2, -0.28], [0, -np.pi / 2, 0]],
            [[0.66, -0.2, -0.0], [0, -np.pi / 2, 0]],
            # reachable poses with unconstrained mode
            [[0.30, -0.2, -0.28], [0.0, 0.0, np.pi / 3]],  # top grasp
            # unreachable poses
            [[0.0, -0.85, -0.0], [-np.pi / 2, 0, 0]],  # backwards limit
            [[0.0, -0.58, -0.28], [-np.pi / 2, -np.pi / 2, 0]],  # backwards limit
            [[0.15, 0.35, -0.10], [np.pi / 3, -np.pi / 2, 0]],  # shoulder limit
            [[0.10, 0.20, -0.22], [np.pi / 3, -np.pi / 2, 0]],  # shoulder limit
            [[0.0, -0.2, -0.66], [0.0, 0.0, -np.pi / 3]],  # backwards limit
            [[0.001, -0.2, -0.68], [0.0, 0.0, -np.pi / 3]],  # pose out of reach
            [[0.001, -0.2, -0.659], [0.0, np.pi / 2, 0.0]],  # wrist out of reach
            [[0.38, -0.2, -0.28], [0.0, np.pi / 2, 0.0]],  # wrist limit
            [[0.1, -0.2, 0.0], [0.0, np.pi, 0.0]],  # elbow limit
            [[0.38, -0.2, -0.28], [0.0, 0.0, 0.0]],  # shoulder limit?
            [[0.1, 0.2, -0.1], [0.0, -np.pi / 2, np.pi / 2]],  # shoulder limit
            [[0.0, -0.2, -0.66], [0, 0, 0]],  # backwards limit
        ]
    )
    l_goal_poses = np.array(
        [
            # reachable poses
            [[0.0001, 0.2, -0.6599], [0, 0, 0]],
            [[0.38, 0.2, -0.28], [0, -np.pi / 2, 0]],
            [[0.66, 0.2, -0.0], [0, -np.pi / 2, 0]],
            # reachable poses with unconstrained mode
            [[0.30, 0.2, -0.28], [0.0, 0.0, -np.pi / 3]],  # top grasp
            # unreachable poses
            [[0.0, 0.85, -0.0], [np.pi / 2, 0, 0]],  # backwards limit
            [[0.0, 0.58, -0.28], [np.pi / 2, -np.pi / 2, 0]],  # backwards limit
            [[0.15, -0.35, -0.10], [-np.pi / 3, -np.pi / 2, 0]],  # shoulder limit
            [[0.10, -0.20, -0.22], [-np.pi / 3, -np.pi / 2, 0]],  # shoulder limit
            [[0.0, 0.2, -0.66], [0.0, 0.0, np.pi / 3]],  # backwards limit
            [[0.001, 0.2, -0.68], [0.0, 0.0, np.pi / 3]],  # pose out of reach
            [[0.001, 0.2, -0.659], [0.0, np.pi / 2, 0.0]],  # wrist out of reach
            [[0.38, 0.2, -0.28], [0.0, np.pi / 2, 0.0]],  # wrist limit
            [[0.1, 0.2, 0.0], [0.0, np.pi, 0.0]],  # elbow limit
            [[0.38, 0.2, -0.28], [0.0, 0.0, 0.0]],  # shoulder limit?
            [[0.1, -0.2, -0.1], [0.0, -np.pi / 2, -np.pi / 2]],  # shoulder limit
            [[0.0, 0.2, -0.66], [0, 0, 0]],  # backwards limit
        ]
    )

    for r_goal_pose in r_goal_poses:
        # print(f"Goal pose {r_goal_pose}")
        # is_reachable, interval, get_joints, _ = r_symbolic_ik.is_reachable(r_goal_pose)
        # print(f"Is reachable {is_reachable}")
        print(f"Goal pose {r_goal_pose}")
        rotation_matrix = R.from_euler("xyz", r_goal_pose[1]).as_matrix()
        goal_pose = make_homogenous_matrix_from_rotation_matrix(r_goal_pose[0], rotation_matrix)
        go_to_pose(reachy, goal_pose, "r_arm", controle_type=controle_type, constrained_mode=constrained_mode)
        time.sleep(2.0)

    for l_goal_pose in l_goal_poses:
        # is_reachable, interval, get_joints, _ = l_symbolic_ik.is_reachable(l_goal_pose)
        # print(f"Is reachable {is_reachable}")
        print(f"Goal pose {r_goal_pose}")
        rotation_matrix = R.from_euler("xyz", l_goal_pose[1]).as_matrix()
        goal_pose = make_homogenous_matrix_from_rotation_matrix(l_goal_pose[0], rotation_matrix)
        go_to_pose(reachy, goal_pose, "l_arm", controle_type=controle_type, constrained_mode=constrained_mode)
        time.sleep(2.0)


def null_space_test() -> None:
    print("Trying to connect on localhost Reachy...")
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    # symbolic_ik_r = SymbolicIK(arm="r_arm")
    symbolic_ik_l = SymbolicIK(arm="l_arm")

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
    pose = np.array([goal_position, goal_orientation])
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

    symbolic_ik_r = SymbolicIK()
    symbolic_ik_l = SymbolicIK(arm="l_arm")

    # --------------- Test poses ---------------

    print(" ----- Testing poses ----- \n")

    print("Testing continuous control")
    test_poses(reachy, symbolic_ik_r, symbolic_ik_l, controle_type="continuous", constrained_mode="low_elbow")
    time.sleep(3.0)

    print("Testing discrete control")
    test_poses(reachy, symbolic_ik_r, symbolic_ik_l, controle_type="discrete", constrained_mode="unconstrained")
    time.sleep(3.0)

    # --------------- Go to pose ---------------

    print("----- Go to pose ----- \n")
    r_goal_pose = np.array([[0.0, -0.2, -0.65], [0, 0, 0]])
    l_goal_pose = np.array([[0.0, 0.2, -0.65], [0, 0, 0]])
    r_M = make_homogenous_matrix_from_rotation_matrix(r_goal_pose[0], R.from_euler("xyz", r_goal_pose[1]).as_matrix())
    l_M = make_homogenous_matrix_from_rotation_matrix(l_goal_pose[0], R.from_euler("xyz", l_goal_pose[1]).as_matrix())
    go_to_pose(reachy, r_M, "r_arm")
    go_to_pose(reachy, l_M, "l_arm")
    time.sleep(5.0)

    # ------ Go to pose with choosen theta -----

    print("----- Go to pose with choosen theta ----- \n")
    r_goal_pose = np.array([[0.55, -0.2, 0.0], [0, -np.pi / 2, 0]])
    l_goal_pose = np.array([[0.55, 0.2, 0.0], [0, -np.pi / 2, 0]])
    go_to_pose_with_choosen_theta(reachy, symbolic_ik_r, r_goal_pose, -4 * np.pi / 6, "r_arm")
    go_to_pose_with_choosen_theta(reachy, symbolic_ik_l, l_goal_pose, -2 * np.pi / 6, "l_arm")
    time.sleep(5.0)

    # ------ Go to pose with all theta -----

    print("----- Go to pose with all theta ----- \n")
    goal_pose = np.array([[0.55, -0.2, -0.0], [0, -np.pi / 2, 0]])
    go_to_pose_with_all_theta(reachy, symbolic_ik_r, goal_pose, "r_arm")
    time.sleep(5.0)

    # ------ Go to joints positions --------

    print("----- Go to joints positions ----- \n")
    r_joints = np.array([0, -10, -15, 0, 0, 0, 0])
    l_joints = np.array([0, 10, 15, 0, 0, 0, 0])
    go_to_joint_positions(reachy, r_joints, "r_arm")
    go_to_joint_positions(reachy, l_joints, "l_arm")
    time.sleep(5.0)

    # --------------------------------------

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


if __name__ == "__main__":
    main_test()
    # null_space_test()
