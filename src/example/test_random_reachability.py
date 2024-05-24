import time

import numpy as np
import numpy.typing as npt
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation


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


def random_trajectoy(reachy: ReachySDK) -> None:
    q = [0, 0, 0, 0, 0, 0, 0]
    ik_r = q
    ik_l = q
    q0 = [-65, -45, 30, 0, 0, 0, 0]
    q_amps = [30, 30, 30, 30, 25, 25, 25]

    freq_reductor = 2.0
    freq = [0.3 * freq_reductor, 0.17 * freq_reductor, 0.39 * freq_reductor, 0.18, 0.31, 0.47, 0.25]
    control_freq = 120
    t_init = time.time()
    while True:
        t = time.time() - t_init + 11
        # randomise the joint angles
        r_q = [q0[i] + q_amps[i] * np.sin(2 * np.pi * freq[i] * t) for i in range(7)]
        l_q = [r_q[0], -r_q[1], -r_q[2], r_q[3], -r_q[4], r_q[5], -r_q[6]]

        M_r = reachy.r_arm.forward_kinematics(r_q)
        M_l = reachy.l_arm.forward_kinematics(l_q)

        t0 = time.time()
        try:
            ik_r = reachy.r_arm.inverse_kinematics(M_r)
        except:
            print("Failed to calculate IK for right arm, this should not happen!")
            # raise ValueError("Failed to calculate IK for right arm, this should not happen!")
        t1 = time.time()
        try:
            ik_l = reachy.l_arm.inverse_kinematics(M_l)
        except:
            print("Failed to calculate IK for left arm, this should not happen!")
            # raise ValueError("Failed to calculate IK for left arm, this should not happen!")

        t2 = time.time()
        # print(f" x,y,z, {x,y,z}, roll,pitch,yaw {roll,pitch,yaw}")

        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik_r):
            joint.goal_position = goal_pos

        for joint, goal_pos in zip(reachy.l_arm.joints.values(), ik_l):
            joint.goal_position = goal_pos

        # Testing the symmetry
        l_mod = np.array([ik_l[0], -ik_l[1], -ik_l[2], ik_l[3], -ik_l[4], ik_l[5], -ik_l[6]])
        # calculate l2 distance between r_joints and l_mod
        l2_dist = np.linalg.norm(ik_r - l_mod)
        print(f"l2_dist: {l2_dist:.5f}")
        # print(f"ik_r: {ik_r}")

        reachy.r_arm.forward_kinematics(ik_r)
        r_goal_diff = np.linalg.norm(reachy.r_arm.forward_kinematics(ik_r) - M_r)
        print(f"r_goal_diff: {r_goal_diff:.3f}")
        reachy.l_arm.forward_kinematics(ik_l)
        l_goal_diff = np.linalg.norm(reachy.l_arm.forward_kinematics(ik_l) - M_l)
        print(f"l_goal_diff: {l_goal_diff:.3f}")
        if r_goal_diff < 0.01 and l_goal_diff < 0.01:
            print("precisions OK")
        else:
            print("precisions NOT OK!!")
            print(f"ik_r {np.round(ik_r, 3).tolist()}")
            print(f"ik_l {np.round(ik_l, 3).tolist()}")
            print(f"M_r {M_r}")
            print(f"M_l {M_l}")
            # break
        # print(f"ik_l: {ik_l}")
        if l2_dist < 0.01:
            print("Symmetry OK")
        else:
            print("Symmetry NOT OK!!")
            print(f"ik_r {np.round(ik_r, 3).tolist()}")
            print(f"ik_l_sym {np.round(l_mod, 3).tolist()}")
            print(f"M_r {M_r}")
            print(f"M_l {M_l}")
            break

        # print(f"ik_r: {ik_r}, ik_l: {ik_l}, time_r: {t1-t0}, time_l: {t2-t1}")
        print(f"time_r: {(t1-t0)*1000:.1f}ms\ntime_l: {(t2-t1)*1000:.1f}ms")
        # Trying to emulate a control loop
        time.sleep(max(0, 1.0 / control_freq - (time.time() - t)))


def main_test() -> None:
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()
    print("Putting each joint at 0 degrees angle")
    time.sleep(0.5)
    for joint in reachy.joints.values():
        joint.goal_position = 0
    time.sleep(1.0)

    random_trajectoy(reachy)

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


if __name__ == "__main__":
    main_test()
