import time
from typing import Tuple

import numpy as np
import numpy.typing as npt
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation

from reachy2_symbolic_ik.control_ik import ControlIK
from reachy2_symbolic_ik.symbolic_ik import SymbolicIK


def get_homogeneous_matrix_msg_from_euler(
    position: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),  # (x, y, z)
    euler_angles: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),  # (roll, pitch, yaw)
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = Rotation.from_euler("xyz", euler_angles, degrees=degrees).as_matrix()
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix


def angle_diff(a: float, b: float) -> float:
    """Returns the smallest distance between 2 angles in rads"""
    d = a - b
    d = ((d + np.pi) % (2 * np.pi)) - np.pi
    return d


def set_joints(reachy: ReachySDK, joints: list[float], arm: str) -> None:
    if arm == "r_arm":
        for joint, goal_pos in zip(reachy.r_arm.joints.values(), joints):
            joint.goal_position = goal_pos
    elif arm == "l_arm":
        for joint, goal_pos in zip(reachy.l_arm.joints.values(), joints):
            joint.goal_position = goal_pos
    reachy.send_goal_positions()


def get_ik(
    reachy: ReachySDK, control_ik: ControlIK, M: npt.NDArray[np.float64], arm: str
) -> Tuple[list[float], npt.NDArray[np.float64]]:
    # control_ik = ControlIK()
    joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elbow_position = np.array([0.0, 0.0, 0.0])
    if arm == "r_arm":
        joints = reachy.r_arm.inverse_kinematics(M)
    elif arm == "l_arm":
        joints = reachy.l_arm.inverse_kinematics(M)
    return joints, elbow_position


def test_joints_space(reachy: ReachySDK) -> None:
    q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ik_r = q
    ik_l = q
    q0 = [-45.0, -60.0, 0.0, -45.0, 0.0, 0.0, 0.0]  # ?? Shouldn't it be -90 for the wrist pitch? Why -45?
    q_amps = [30.0, 30.0, 30.0, 45.0, 25.0, 25.0, 90.0]

    control_ik = ControlIK()
    symbolik_r = SymbolicIK(arm="r_arm")
    symbolik_l = SymbolicIK(arm="l_arm")

    freq_reductor = 2.0
    freq = [0.3 * freq_reductor, 0.17 * freq_reductor, 0.39 * freq_reductor, 0.18, 0.31, 0.47, 0.25]
    # control_freq = 120
    t_init = time.time()
    while True:
        # input("go?")
        t = time.time() - t_init + 11

        r_q = [q0[i] + q_amps[i] * np.sin(2 * np.pi * freq[i] * t) for i in range(7)]

        # l_q = [r_q[0], -r_q[1], -r_q[2], r_q[3], -r_q[4], r_q[5], -r_q[6]]
        M_r = reachy.r_arm.forward_kinematics(r_q)
        # M_l = reachy.l_arm.forward_kinematics(l_q)
        M_l = np.array(
            [
                [M_r[0][0], -M_r[0][1], M_r[0][2], M_r[0][3]],
                [-M_r[1][0], M_r[1][1], -M_r[1][2], -M_r[1][3]],
                [M_r[2][0], -M_r[2][1], M_r[2][2], M_r[2][3]],
                [0, 0, 0, 1],
            ]
        )

        r_goal_position = M_r[:3, 3]
        r_goal_orientation = Rotation.from_matrix(M_r[:3, :3]).as_euler("xyz", degrees=False)
        r_goal_pose = np.array([r_goal_position, r_goal_orientation])

        l_goal_position = M_l[:3, 3]
        l_goal_orientation = Rotation.from_matrix(M_l[:3, :3]).as_euler("xyz", degrees=False)
        l_goal_pose = np.array([l_goal_position, l_goal_orientation])

        is_reachable_r, _, _, state = symbolik_r.is_reachable(r_goal_pose)
        is_reachable_l, _, _, state = symbolik_l.is_reachable(l_goal_pose)
        if not is_reachable_r:
            print(" ##### Target is not reachable by r arms #####")
            print(f"state: {state}")
            print(f"r_q: {r_q}")
            print(f"M_r: {M_r}")
            print(f"goal_pose: {r_goal_pose}")
            break
        if not is_reachable_l:
            print(" ##### Target is not reachable by l arms #####")
            print(f"state: {state}")
            print(f"r_q: {r_q}")
            print(f"M_r: {M_r}")
            print(f"goal_pose: {l_goal_pose}")
            break

        t0 = time.time()
        try:
            ik_r, elbow_position_r = get_ik(reachy, control_ik, M_r, "r_arm")
            # ik_r = reachy.r_arm.inverse_kinematics(M_r)
        except Exception as e:
            # print("Failed to calculate IK for right arm, this should not happen!")
            raise ValueError(f"Failed to calculate IK for right arm, this should not happen! {e}")
        t1 = time.time()
        try:
            ik_l, elbow_position_l = get_ik(reachy, control_ik, M_l, "l_arm")
            # ik_l = reachy.l_arm.inverse_kinematics(M_l)
        except Exception as e:
            # print("Failed to calculate IK for left arm, this should not happen!")
            raise ValueError(f"Failed to calculate IK for left arm, this should not happen! {e}")
        t2 = time.time()
        print(f"time_r: {(t1-t0)*1000:.1f}ms\ntime_l: {(t2-t1)*1000:.1f}ms")

        set_joints(reachy, ik_r, "r_arm")
        set_joints(reachy, ik_l, "l_arm")


if __name__ == "__main__":
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
    reachy.turn_on()
    print("Putting each joint at 0 degrees angle")
    time.sleep(0.5)
    for joint in reachy.joints.values():
        joint.goal_position = 0
    time.sleep(1.0)
    test_joints_space(reachy)
