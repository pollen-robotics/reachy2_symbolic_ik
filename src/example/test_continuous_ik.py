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


def random_trajectoy(reachy: ReachySDK) -> None: # noqa: C901
    x0, y0, z0 = 0.3, -0.4, -0.3
    roll0, pitch0, yaw0 = 0.0, -np.pi / 2, 0.0
    reductor = 1.0
    freq_reductor = 2.0
    # Note Orbita3D's cone is currently limited to 42.5 rad in the control loop, 
    # but this can't be represented correctly with the xyz euler angles used here
    amp = [0.35 * reductor, 0.35 * reductor, 0.35 * reductor, np.pi / 6, np.pi / 6, np.pi / 6]
    freq = [0.3 * freq_reductor, 0.17 * freq_reductor, 0.39 * freq_reductor, 0.18, 0.31, 0.47]
    control_freq = 120
    max_angular_change = 5.0  # degrees
    ik_r = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ik_l = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    prev_ik_r = ik_r
    prev_ik_l = ik_l
    prev_M_r = get_homogeneous_matrix_msg_from_euler(np.array([x0, y0, z0]), np.array([roll0, pitch0, yaw0]), degrees=False)
    prev_M_l = get_homogeneous_matrix_msg_from_euler(np.array([x0, -y0, z0]), np.array([-roll0, pitch0, -yaw0]), degrees=False)
    first = True
    t_init = time.time()
    while True:
        # input("go?")
        t = time.time() - t_init + 11
        x = x0 + amp[0] * np.sin(freq[0] * t)
        y = y0 + amp[1] * np.sin(freq[1] * t)
        z = z0 + amp[2] * np.sin(freq[2] * t)
        roll = roll0 + amp[3] * np.sin(freq[3] * t)
        pitch = pitch0 + amp[4] * np.sin(freq[4] * t)
        yaw = yaw0 + amp[5] * np.sin(freq[5] * t)

        M_r = get_homogeneous_matrix_msg_from_euler(np.array([x, y, z]), np.array([roll, pitch, yaw]), degrees=False)
        M_l = get_homogeneous_matrix_msg_from_euler(np.array([x, -y, z]), np.array([-roll, pitch, -yaw]), degrees=False)

        t0 = time.time()
        try:
            ik_r = reachy.r_arm.inverse_kinematics(M_r)
        except Exception as e:
            print("Failed to calculate IK for right arm, this should not happen!")
            # raise ValueError("Failed to calculate IK for right arm, this should not happen!")
        t1 = time.time()
        try:
            ik_l = reachy.l_arm.inverse_kinematics(M_l)
        except Exception as e:
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
        print(f"l2_dist: {l2_dist:.3f}")
        # print(f"ik_r: {ik_r}")
        reachy.r_arm.forward_kinematics(ik_r)
        goal_diff = np.linalg.norm(reachy.r_arm.forward_kinematics(ik_r) - M_r)
        print(f"goal_diff: {goal_diff:.3f}")
        # print(f"ik_l: {ik_l}")
        if l2_dist < 0.0001:
            print("Symmetry OK")
            pass
        else:
            pass
            print("Symmetry NOT OK!!")
            print(f"prev_ik_r {np.round(prev_ik_r, 3).tolist()}")
            print(f"prev_ik_l {np.round(prev_ik_l, 3).tolist()}")
            print(f"ik_r {np.round(ik_r, 3).tolist()}")
            print(f"ik_l_sym {np.round(l_mod, 3).tolist()}")
            print(f"prev_M_r {prev_M_r}")
            print(f"prev_M_l {prev_M_l}")
            print(f"M_r {M_r}")
            print(f"M_l {M_l}")
            break

        # Test continuity
        # calculating the maximum angulare change in joint space
        # create a list based on angle_diff for each joint
        r_diff = [angle_diff(np.deg2rad(a), np.deg2rad(b)) for a, b in zip(ik_r, prev_ik_r)]
        l_diff = [angle_diff(np.deg2rad(a), np.deg2rad(b)) for a, b in zip(ik_l, prev_ik_l)]
        max_angular_change_r = np.rad2deg(np.max(np.abs(r_diff)))
        max_angular_change_l = np.rad2deg(np.max(np.abs(l_diff)))

        if max_angular_change_r < max_angular_change and max_angular_change_l < max_angular_change:
            print("Continuity OK")
        else:
            print("Continuity NOT OK!!")
            print(f"prev_ik_r {np.round(prev_ik_r, 3).tolist()}")
            print(f"prev_ik_l {np.round(prev_ik_l, 3).tolist()}")
            print(f"ik_r {np.round(ik_r, 3).tolist()}")
            print(f"ik_l {np.round(ik_l, 3).tolist()}")
            print(f"max_angular_change_r {max_angular_change_r:.3f}")
            print(f"max_angular_change_l {max_angular_change_l:.3f}")
            print(f"prev_M_r {prev_M_r}")
            print(f"prev_M_l {prev_M_l}")
            print(f"M_r {M_r}")
            print(f"M_l {M_l}")
            if not first:
                break

        prev_ik_r = ik_r
        prev_ik_l = ik_l
        prev_M_r = M_r
        prev_M_l = M_l
        first = False

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
