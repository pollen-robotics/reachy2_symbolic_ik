import time
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
from reachy2_sdk.reachy_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.utils import make_homogenous_matrix_from_rotation_matrix


def dk(joints: npt.NDArray[np.float64]) -> Any:
    shoulder_offset = np.array([10, 0, 15])
    shoulder_position = [0, -0.2, 0]
    upper_arm_size = 0.28
    forearm_size = 0.28
    gripper_size = 0.1
    shoulder_offset = np.radians(shoulder_offset)
    M_shoulder_offset = R.from_euler("xyz", shoulder_offset).as_matrix()
    M_shoulder_pitch = R.from_euler("xyz", [0, joints[0], 0]).as_matrix()
    M_shoulder_roll = R.from_euler("xyz", [joints[1], 0, 0]).as_matrix()
    M_torso_shoulder = np.dot(M_shoulder_offset, M_shoulder_pitch)
    M_torso_shoulder = np.dot(M_torso_shoulder, M_shoulder_roll)
    T_torso_shoulder = make_homogenous_matrix_from_rotation_matrix(shoulder_position, M_torso_shoulder)
    elbow_position = [0, 0, -upper_arm_size, 1]
    P_torso_elbow = np.dot(T_torso_shoulder, elbow_position)
    # print(f"elbow_position: {P_torso_elbow}")
    M_elbow_yaw = R.from_euler("xyz", [0, 0, joints[2]]).as_matrix()
    M_elbow_pitch = R.from_euler("xyz", [0, joints[3], 0]).as_matrix()
    M_elbow = np.dot(M_elbow_yaw, M_elbow_pitch)
    M_torso_elbow = np.dot(M_torso_shoulder, M_elbow)
    T_torso__elbow = make_homogenous_matrix_from_rotation_matrix(P_torso_elbow[:3], M_torso_elbow)
    wrist_position = [0, 0, -forearm_size, 1]
    P_torso_wrist = np.dot(T_torso__elbow, wrist_position)
    # print(f"wrist_position: {P_torso_wrist}")
    M_wrist_roll = R.from_euler("xyz", [joints[4], 0, 0]).as_matrix()
    M_wrist_pitch = R.from_euler("xyz", [0, joints[5], 0]).as_matrix()
    M_wrist_yaw = R.from_euler("xyz", [0, 0, joints[6]]).as_matrix()
    M_wrist = np.dot(M_wrist_roll, M_wrist_pitch)
    M_wrist = np.dot(M_wrist, M_wrist_yaw)
    M_torso_wrist = np.dot(M_torso_elbow, M_wrist)
    T_torso_wrist = make_homogenous_matrix_from_rotation_matrix(P_torso_wrist[:3], M_torso_wrist)
    tip_position = [0, 0, -gripper_size, 1]
    P_torso_tip = np.dot(T_torso_wrist, tip_position)
    T_torso_tip = make_homogenous_matrix_from_rotation_matrix(P_torso_tip[:3], M_torso_wrist)
    # print(f"tip_position: {P_torso_tip}")
    return T_torso_tip


def get_distance_between_poses(
    goal_pose: npt.NDArray[np.float64], current_pose: npt.NDArray[np.float64]
) -> Tuple[np.float64, np.float64]:
    l2 = np.linalg.norm(goal_pose[:3, 3] - current_pose[:3, 3])
    R_goal_pose = goal_pose[:3, :3]
    R_current_pose = current_pose[:3, :3]
    R_diff = np.dot(R_goal_pose, R_current_pose.T)
    axis_angle = R.from_matrix(R_diff).as_rotvec()
    angle = np.linalg.norm(axis_angle)
    return angle, l2


def main() -> None:
    reachy = ReachySDK(host="localhost")
    reachy.turn_on()
    time.sleep(1)

    # goal_position = [0.38, -0.2, -0.28]
    # goal_orientation = [0, -np.pi/2, 0]
    # goal_position = [0.001, -0.2, -0.6599]
    # goal_orientation = [0, 0, 0]
    # goal_position = [0.45, -0.2, -0.]
    # goal_orientation = [-np.pi/4, -np.pi/2, 0]
    goal_position = [0.66, -0.2, -0.0]
    goal_orientation = [0, -np.pi / 2, 0]

    target_pose = make_homogenous_matrix_from_rotation_matrix(goal_position, R.from_euler("xyz", goal_orientation).as_matrix())
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
        joint.goal_position = goal_pos
    reachy.send_goal_positions()

    print(f"target_pose, {target_pose}")
    print(f"goal joints{ik}\n")

    print("-- KDL FK --")
    t = time.time()
    for i in range(100):
        current_pos = reachy.r_arm.forward_kinematics()
    print(f"Time: {(time.time() - t)*10}ms")
    print(f"current_pos {current_pos}")
    # print(f" current_pos - target_pose {current_pos - target_pose}")
    # print(np.isclose(current_pos, target_pose, atol=1e-05))
    angle, l2 = get_distance_between_poses(target_pose, current_pos)
    print(f"angle: {angle} (rad)")
    print(f"l2: {l2} (m)\n")

    print("-- Symbolic FK --")
    t = time.time()
    for i in range(100):
        T_torso_tip = dk(np.radians(ik))
    print(f"Time: {(time.time() - t)*10}ms")
    print(f"current_pos {T_torso_tip}")
    # print(f" current_pos - target_pose {T_torso_tip - target_pose}")
    # print(np.isclose(T_torso_tip, target_pose, atol=1e-05))
    angle, l2 = get_distance_between_poses(target_pose, T_torso_tip)
    print(f"angle: {angle} (rad)")
    print(f"l2: {l2} (m)")


if __name__ == "__main__":
    main()
