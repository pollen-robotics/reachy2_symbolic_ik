import time
from cmath import phase
from re import M

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def get_homogeneous_matrix_msg_from_euler(
    position: tuple = (0, 0, 0),  # (x, y, z)
    euler_angles: tuple = (0, 0, 0),  # (roll, pitch, yaw)
    degrees: bool = False,
):
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = Rotation.from_euler("xyz", euler_angles, degrees=degrees).as_matrix()
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix


def angle_diff(a: float, b: float) -> float:
    """Returns the smallest distance between 2 angles"""
    d = a - b
    d = ((d + np.pi) % (2 * np.pi)) - np.pi
    return d


@pytest.mark.sdk
def test_continuous_ik():
    from reachy2_sdk import ReachySDK

    reachy = ReachySDK(host="localhost")
    assert not (reachy._grpc_status == "disconnected")
    time.sleep(0.5)
    reachy.turn_on()
    time.sleep(0.5)

    x0, y0, z0 = 0.65, -0.2, -0.0
    roll0, pitch0, yaw0 = 0, -np.pi / 2, 0
    reductor = 1.0
    freq_reductor = 2.0
    amp = [0.35 * reductor, 0.35 * reductor, 0.35 * reductor, np.pi / 6, np.pi / 6, np.pi / 6]
    freq = [0.3 * freq_reductor, 0.17 * freq_reductor, 0.39 * freq_reductor, 0.18, 0.31, 0.47]
    control_freq = 120
    max_angular_change = 5.0  # degrees
    prev_ik_r = [0, -10, -15, 0, 0, 0, 0]
    prev_ik_l = [0, 10, 15, 0, 0, 0, 0]
    # prev_M_r = get_homogeneous_matrix_msg_from_euler((x0, y0, z0), (roll0, pitch0, yaw0), degrees=False)
    # prev_M_l = get_homogeneous_matrix_msg_from_euler((x0, -y0, z0), (-roll0, pitch0, -yaw0), degrees=False)
    # first = True

    # TODO fix this test : continuous test can not work with the discrete control function

    #
    #     # t_init = time.time()
    # while time.time() - t_init < 10:
    #     t = time.time() - t_init + 11
    #     x = x0 + amp[0] * np.sin(freq[0] * t)
    #     y = y0 + amp[1] * np.sin(freq[1] * t)
    #     z = z0 + amp[2] * np.sin(freq[2] * t)
    #     roll = roll0 + amp[3] * np.sin(freq[3] * t)
    #     pitch = pitch0 + amp[4] * np.sin(freq[4] * t)
    #     yaw = yaw0 + amp[5] * np.sin(freq[5] * t)

    #     M_r = get_homogeneous_matrix_msg_from_euler((x, y, z), (roll, pitch, yaw), degrees=False)
    #     M_l = get_homogeneous_matrix_msg_from_euler((x, -y, z), (-roll, pitch, -yaw), degrees=False)

    #     # t0 = time.time()

    #     ik_r = reachy.r_arm.inverse_kinematics(M_r)
    #     # t1 = time.time()
    #     ik_l = reachy.l_arm.inverse_kinematics(M_l)
    #     # t2 = time.time()
    #     # print(f" x,y,z, {x,y,z}, roll,pitch,yaw {roll,pitch,yaw}")

    #     for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik_r):
    #         joint.goal_position = goal_pos

    #     for joint, goal_pos in zip(reachy.l_arm.joints.values(), ik_l):
    #         joint.goal_position = goal_pos

    #     # Testing the symmetry
    #     l_mod = np.array([ik_l[0], -ik_l[1], -ik_l[2], ik_l[3], -ik_l[4], ik_l[5], -ik_l[6]])
    #     # calculate l2 distance between r_joints and l_mod
    #     l2_dist = np.linalg.norm(ik_r - l_mod)
    #     # print(f"l2_dist: {l2_dist:.3f}")
    #     # print(f"ik_r: {ik_r}")
    #     # print(f"ik_l: {ik_l}")
    #     # assert l2_dist < 1e-8

    #     # if l2_dist < 0.0001:
    #     #     print("Symmetry OK")
    #     #     pass
    #     # else:
    #     #     pass
    #     #     print("Symmetry NOT OK!!")
    #     #     print(f"prev_ik_r {np.round(prev_ik_r, 3).tolist()}")
    #     #     print(f"prev_ik_l {np.round(prev_ik_l, 3).tolist()}")
    #     #     print(f"ik_r {np.round(ik_r, 3).tolist()}")
    #     #     print(f"ik_l_sym {np.round(l_mod, 3).tolist()}")
    #     #     print(f"prev_M_r {prev_M_r}")
    #     #     print(f"prev_M_l {prev_M_l}")
    #     #     print(f"M_r {M_r}")
    #     #     print(f"M_l {M_l}")
    #     #     break

    #     # Test continuity
    #     # calculating the maximum angulare change in joint space
    #     # create a list based on angle_diff for each joint
    #     r_diff = [angle_diff(a, b) for a, b in zip(ik_r, prev_ik_r)]
    #     l_diff = [angle_diff(a, b) for a, b in zip(ik_l, prev_ik_l)]
    #     max_angular_change_r = np.max(np.abs(r_diff))
    #     max_angular_change_l = np.max(np.abs(l_diff))
    #     # assert max_angular_change_r < max_angular_change
    #     # assert max_angular_change_l < max_angular_change

    #     # if max_angular_change_r < max_angular_change and max_angular_change_l < max_angular_change:
    #     #     print("Continuity OK")
    #     # else:
    #     #     print("Continuity NOT OK!!")
    #     #     print(f"prev_ik_r {np.round(prev_ik_r, 3).tolist()}")
    #     #     print(f"prev_ik_l {np.round(prev_ik_l, 3).tolist()}")
    #     #     print(f"ik_r {np.round(ik_r, 3).tolist()}")
    #     #     print(f"ik_l {np.round(ik_l, 3).tolist()}")
    #     #     print(f"max_angular_change_r {max_angular_change_r:.3f}")
    #     #     print(f"max_angular_change_l {max_angular_change_l:.3f}")
    #     #     print(f"prev_M_r {prev_M_r}")
    #     #     print(f"prev_M_l {prev_M_l}")
    #     #     print(f"M_r {M_r}")
    #     #     print(f"M_l {M_l}")
    #     #     if not first:
    #     #         break

    #     prev_ik_r = ik_r
    #     prev_ik_l = ik_l
    #     # prev_M_r = M_r
    #     # prev_M_l = M_l
    #     # first = False

    #     # print(f"ik_r: {ik_r}, ik_l: {ik_l}, time_r: {t1-t0}, time_l: {t2-t1}")
    #     # print(f"time_r: {(t1-t0)*1000:.1f}ms\ntime_l: {(t2-t1)*1000:.1f}ms")
    #     # Trying to emulate a control loop
    #     time.sleep(max(0, 1.0 / control_freq - (time.time() - t)))
    # reachy.disconnect()
