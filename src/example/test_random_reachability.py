import time
from typing import Tuple

import numpy as np
import numpy.typing as npt
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value
from reachy2_sdk import ReachySDK
from reachy2_sdk_api.arm_pb2 import (
    ArmCartesianGoal,
    IKConstrainedMode,
    IKContinuousMode,
)
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4
from scipy.spatial.transform import Rotation

from reachy2_symbolic_ik.control_ik import ControlIK
from reachy2_symbolic_ik.utils import distance_from_singularity

# CONTROLE_TYPE = "local_discrete"
# CONTROLE_TYPE = "local_continuous"
CONTROLE_TYPE = "sdk_discrete"
# CONTROLE_TYPE = "sdk_continuous"


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
    if CONTROLE_TYPE == "local_discrete":
        joints, is_reachable, state = control_ik.symbolic_inverse_kinematics(arm, M, "discrete")
        joints = list(np.degrees(joints))
        elbow_position = (control_ik.symbolic_ik_solver[arm].elbow_position)[:3]
        # print(f"elbow_position: {elbow_position}")
    elif CONTROLE_TYPE == "local_continuous":
        joints, is_reachable, state = control_ik.symbolic_inverse_kinematics(
            arm, M, "continuous", constrained_mode="unconstrained"
        )
        joints = list(np.degrees(joints))
        elbow_position = (control_ik.symbolic_ik_solver[arm].elbow_position)[:3]
        # print(f"joint angles: {joints}")
        # print(f"is_reachable: {is_reachable}, state: {state}")
    elif CONTROLE_TYPE == "sdk_discrete":
        if arm == "r_arm":
            joints = reachy.r_arm.inverse_kinematics(M)
        elif arm == "l_arm":
            joints = reachy.l_arm.inverse_kinematics(M)
    elif CONTROLE_TYPE == "sdk_continuous":
        if arm == "r_arm":
            request = ArmCartesianGoal(
                id=reachy.r_arm._part_id,
                goal_pose=Matrix4x4(data=M.flatten().tolist()),
                continuous_mode=IKContinuousMode.CONTINUOUS,
                constrained_mode=IKConstrainedMode.UNCONSTRAINED,
                preferred_theta=FloatValue(
                    value=-4 * np.pi / 6,
                ),
                d_theta_max=FloatValue(value=0.01),
                order_id=Int32Value(value=5),
            )
            reachy.r_arm._arm_stub.SendArmCartesianGoal(request)
            # print(reachy.r_arm.shoulder.pitch.present_position)
        elif arm == "l_arm":
            request = ArmCartesianGoal(
                id=reachy.l_arm._part_id,
                goal_pose=Matrix4x4(data=M.flatten().tolist()),
                continuous_mode=IKContinuousMode.CONTINUOUS,
                constrained_mode=IKConstrainedMode.LOW_ELBOW,
                preferred_theta=FloatValue(
                    value=-4 * np.pi / 6,
                ),
                d_theta_max=FloatValue(value=0.01),
                order_id=Int32Value(value=5),
            )
            reachy.l_arm._arm_stub.SendArmCartesianGoal(request)
    return joints, elbow_position


def random_trajectoy(reachy: ReachySDK, debug_pose: bool = False, bypass: bool = False) -> None:
    q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ik_r = q
    ik_l = q
    q0 = [-45.0, -60.0, 0.0, -45.0, 0.0, 0.0, 0.0]  # ?? Shouldn't it be -90 for the wrist pitch? Why -45?
    q_amps = [30.0, 30.0, 30.0, 45.0, 25.0, 25.0, 90.0]

    control_ik = ControlIK()

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
        # M_l = reachy.l_arm.forward_kinematics(l_q)
        M_l = np.array(
            [
                [M_r[0][0], -M_r[0][1], M_r[0][2], M_r[0][3]],
                [-M_r[1][0], M_r[1][1], -M_r[1][2], -M_r[1][3]],
                [M_r[2][0], -M_r[2][1], M_r[2][2], M_r[2][3]],
                [0, 0, 0, 1],
            ]
        )

        if not bypass:
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
        else:
            ik_r = r_q
            ik_l = l_q
        # print(f" x,y,z, {x,y,z}, roll,pitch,yaw {roll,pitch,yaw}")

        if CONTROLE_TYPE == "sdk_continuous":
            time.sleep(0.1)
            r_real_pose = reachy.r_arm.forward_kinematics()
            l_real_pose = reachy.l_arm.forward_kinematics()
        else:
            set_joints(reachy, ik_r, "r_arm")
            set_joints(reachy, ik_l, "l_arm")
            r_real_pose = reachy.r_arm.forward_kinematics(ik_r)
            l_real_pose = reachy.l_arm.forward_kinematics(ik_l)

        is_real_pose_correct = check_precision_and_symmetry(
            reachy, M_r, M_l, r_real_pose, l_real_pose, ik_r, ik_l, elbow_position_r, elbow_position_l
        )
        if not is_real_pose_correct:
            break

        # print(f"ik_r: {ik_r}, ik_l: {ik_l}, time_r: {t1-t0}, time_l: {t2-t1}")
        # Trying to emulate a control loop
        # print(max(0, 1.0 / control_freq - (time.time() - t)))
        time.sleep(max(0, 1.0 / control_freq - (time.time() - t)))


def check_precision_and_symmetry(
    reachy: ReachySDK,
    M_r: npt.NDArray[np.float64],
    M_l: npt.NDArray[np.float64],
    r_real_pose: npt.NDArray[np.float64],
    l_real_pose: npt.NDArray[np.float64],
    ik_r: list[float],
    ik_l: list[float],
    elbow_position_r: npt.NDArray[np.float64],
    elbow_position_l: npt.NDArray[np.float64],
) -> bool:
    is_real_pose_correct = True

    if CONTROLE_TYPE == "local_continuous" or CONTROLE_TYPE == "local_discrete":
        # print(elbow_position_r)
        distance_from_singularity_r = distance_from_singularity(elbow_position_r, "r_arm")
        distance_from_singularity_l = distance_from_singularity(elbow_position_l, "l_arm")
        print(f"distance_from_singularity_r: {distance_from_singularity_r:.5f}")
        print(f"distance_from_singularity_l: {distance_from_singularity_l:.5f}")
        if distance_from_singularity_r < 1e-4 or distance_from_singularity_l < 1e-4:
            print("Singularity reached_______________________________________________________________________")

    l_mod = np.array([ik_l[0], -ik_l[1], -ik_l[2], ik_l[3], -ik_l[4], ik_l[5], -ik_l[6]])
    # calculate l2 distance between r_joints and l_mod
    l2_dist = np.linalg.norm(ik_r - l_mod)
    print(f"l2_dist: {l2_dist:.5f}")

    r_position_diff = np.linalg.norm(r_real_pose[:3, 3] - M_r[:3, 3])
    r_euler_diff = np.linalg.norm(
        Rotation.from_matrix(r_real_pose[:3, :3]).as_euler("xyz") - Rotation.from_matrix(M_r[:3, :3]).as_euler("xyz")
    )
    print(f"r_position_diff: {r_position_diff:.3f}")
    print(f"r_euler_diff: {r_euler_diff:.3f}")

    l_position_diff = np.linalg.norm(l_real_pose[:3, 3] - M_l[:3, 3])
    l_euler_diff = np.linalg.norm(
        Rotation.from_matrix(l_real_pose[:3, :3]).as_euler("xyz") - Rotation.from_matrix(M_l[:3, :3]).as_euler("xyz")
    )
    print(f"l_position_diff: {l_position_diff:.3f}")
    print(f"l_euler_diff: {l_euler_diff:.3f}")

    if (r_position_diff < 0.01 or r_position_diff < 0.01) and (l_position_diff < 0.01 or l_position_diff < 0.01):
        print("precisions OK")
    else:
        print("precisions NOT OK!!")
        # print(f"initial r_q {r_q}")
        print(f"real_pose_r {reachy.r_arm.forward_kinematics()}")
        print(f"real_pose_l {reachy.l_arm.forward_kinematics()}")
        print(f"ik_r {np.round(ik_r, 3).tolist()}")
        print(f"ik_l {np.round(ik_l, 3).tolist()}")
        print(f"M_r {M_r}")
        print(f"M_l {M_l}")
        is_real_pose_correct = False
    # print(f"ik_l: {ik_l}")
    if l2_dist < 0.1:
        print("Symmetry OK")
    else:
        print("Symmetry NOT OK!!")
        # print(f"initial r_q {r_q}")
        print(f"ik_r {np.round(ik_r, 3).tolist()}")
        print(f"ik_l_sym {np.round(l_mod, 3).tolist()}")
        print(f"M_r {M_r}")
        print(f"M_l {M_l}")
        is_real_pose_correct = False
    print("_____________________")
    return is_real_pose_correct


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
    reachy.send_goal_positions()


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
