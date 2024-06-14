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


def set_joints(reachy: ReachySDK, joints: list[float], arm: str) -> None:
    if arm == "r_arm":
        for joint, goal_pos in zip(reachy.r_arm.joints.values(), joints):
            joint.goal_position = goal_pos
    elif arm == "l_arm":
        for joint, goal_pos in zip(reachy.l_arm.joints.values(), joints):
            joint.goal_position = goal_pos


def random_trajectoy(reachy: ReachySDK, debug_pose: bool = False, bypass: bool = False) -> None:
    q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ik_r = q
    ik_l = q
    q0 = [-45.0, -60.0, 0.0, -45.0, 0.0, 0.0, 0.0]  # ?? Shouldn't it be -90 for the wrist pitch? Why -45?
    q_amps = [30.0, 30.0, 30.0, 45.0, 25.0, 25.0, 90.0]

    freq_reductor = 2.0
    freq = [0.3 * freq_reductor, 0.17 * freq_reductor, 0.39 * freq_reductor, 0.18, 0.31, 0.47, 0.25]
    control_freq = 120
    t_init = time.time()
    while True:
        # input("go?")
        t = time.time() - t_init + 11
        # randomise the joint angles
        if not debug_pose:
            r_q = [q0[i] + q_amps[i] * np.sin(2 * np.pi * freq[i] * t) for i in range(7)]
        else:
            # r_q = [
            #     -35.78853788564081,
            #     -15.498597138514409,
            #     25.764970562656003,
            #     -29.883534556661207,
            #     -24.94795468578275,
            #     24.47051189664784,
            #     -21.197897851824024,
            # ]
            # r_q = [
            #     -88.382705930125,
            #     -36.21995282072161,
            #     29.280905817285657,
            #     76.24441255844017,
            #     9.179832209607154,
            #     -24.301622376036853,
            #     24.714757308672915,
            # ]
            # r_q = q0
            # Precision problem
            r_q = [
                -41.03096373192293,
                -37.647921777704,
                -29.585523143459014,
                -88.24667866025105,
                -21.052896284656175,
                9.808669854062696,
                -89.89911806297954,
            ]

        l_q = [r_q[0], -r_q[1], -r_q[2], r_q[3], -r_q[4], r_q[5], -r_q[6]]
        M_r = reachy.r_arm.forward_kinematics(r_q)
        M_l = reachy.l_arm.forward_kinematics(l_q)

        if not bypass:
            t0 = time.time()
            try:
                ik_r = reachy.r_arm.inverse_kinematics(M_r)
            except Exception as e:
                # print("Failed to calculate IK for right arm, this should not happen!")
                raise ValueError(f"Failed to calculate IK for right arm, this should not happen! {e}")
            t1 = time.time()
            try:
                ik_l = reachy.l_arm.inverse_kinematics(M_l)
            except Exception as e:
                # print("Failed to calculate IK for left arm, this should not happen!")
                raise ValueError(f"Failed to calculate IK for left arm, this should not happen! {e}")
            t2 = time.time()
            print(f"time_r: {(t1-t0)*1000:.1f}ms\ntime_l: {(t2-t1)*1000:.1f}ms")
        else:
            ik_r = r_q
            ik_l = l_q
        # print(f" x,y,z, {x,y,z}, roll,pitch,yaw {roll,pitch,yaw}")

        set_joints(reachy, ik_r, "r_arm")
        set_joints(reachy, ik_l, "l_arm")

        # for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik_r):
        #     joint.goal_position = goal_pos

        # for joint, goal_pos in zip(reachy.l_arm.joints.values(), ik_l):
        #     joint.goal_position = goal_pos

        # Testing the symmetry
        l_mod = np.array([ik_l[0], -ik_l[1], -ik_l[2], ik_l[3], -ik_l[4], ik_l[5], -ik_l[6]])
        # calculate l2 distance between r_joints and l_mod
        l2_dist = np.linalg.norm(ik_r - l_mod)
        print(f"l2_dist: {l2_dist:.5f}")
        # print(f"ik_r: {ik_r}")

        reachy.r_arm.forward_kinematics(ik_r)
        # r_goal_diff = np.linalg.norm(reachy.r_arm.forward_kinematics(ik_r) - M_r)
        r_position_diff = np.linalg.norm(reachy.r_arm.forward_kinematics(ik_r)[:3, 3] - M_r[:3, 3])
        r_euler_diff = np.linalg.norm(
            Rotation.from_matrix(reachy.r_arm.forward_kinematics(ik_r)[:3, :3]).as_euler("xyz")
            - Rotation.from_matrix(M_r[:3, :3]).as_euler("xyz")
        )
        print(f"r_position_diff: {r_position_diff:.3f}")
        print(f"r_euler_diff: {r_euler_diff:.3f}")

        # print(f"r_goal_diff: {r_goal_diff:.3f}")
        reachy.l_arm.forward_kinematics(ik_l)
        # l_goal_diff = np.linalg.norm(reachy.l_arm.forward_kinematics(ik_l) - M_l)
        l_position_diff = np.linalg.norm(reachy.l_arm.forward_kinematics(ik_l)[:3, 3] - M_l[:3, 3])
        l_euler_diff = np.linalg.norm(
            Rotation.from_matrix(reachy.l_arm.forward_kinematics(ik_l)[:3, :3]).as_euler("xyz")
            - Rotation.from_matrix(M_l[:3, :3]).as_euler("xyz")
        )
        print(f"l_position_diff: {l_position_diff:.3f}")
        print(f"l_euler_diff: {l_euler_diff:.3f}")

        # print(f"l_goal_diff: {l_goal_diff:.3f}")
        if (r_position_diff < 0.01 or r_position_diff < 0.01) and (l_position_diff < 0.01 or l_position_diff < 0.01):
            print("precisions OK")
        else:
            print("precisions NOT OK!!")
            print(f"initial r_q {r_q}")
            print(f"ik_r {np.round(ik_r, 3).tolist()}")
            print(f"ik_l {np.round(ik_l, 3).tolist()}")
            print(f"M_r {M_r}")
            print(f"M_l {M_l}")
            break
        print(f"ik_l: {ik_l}")
        if l2_dist < 0.1:
            print("Symmetry OK")
        else:
            print("Symmetry NOT OK!!")
            print(f"initial r_q {r_q}")
            print(f"ik_r {np.round(ik_r, 3).tolist()}")
            print(f"ik_l_sym {np.round(l_mod, 3).tolist()}")
            print(f"M_r {M_r}")
            print(f"M_l {M_l}")
            break

        # print(f"ik_r: {ik_r}, ik_l: {ik_l}, time_r: {t1-t0}, time_l: {t2-t1}")
        # Trying to emulate a control loop
        time.sleep(max(0, 1.0 / control_freq - (time.time() - t)))


def test_joints(reachy: ReachySDK) -> None:
    # q0 = [-65, -45, 0, 45, 0, -45, 0]  # ?? Shouldn't it be -90 for the wrist pitch? Why -45?

    r_q = [
        -88.382705930125,
        -36.21995282072161,
        29.280905817285657,
        76.24441255844017,
        9.179832209607154,
        -24.301622376036853,
        24.714757308672915,
    ]
    l_q = [r_q[0], -r_q[1], -r_q[2], r_q[3], -r_q[4], r_q[5], -r_q[6]]

    for joint, goal_pos in zip(reachy.r_arm.joints.values(), r_q):
        joint.goal_position = goal_pos

    for joint, goal_pos in zip(reachy.l_arm.joints.values(), l_q):
        joint.goal_position = goal_pos


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

    random_trajectoy(reachy, debug_pose=False, bypass=False)
    # test_joints(reachy)

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


if __name__ == "__main__":
    main_test()
