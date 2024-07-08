import time

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
from reachy2_sdk.utils.utils import recompose_matrix, decompose_matrix
from pyquaternion import Quaternion


def get_homogeneous_matrix_msg_from_euler(
    position: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),  # (x, y, z)
    euler_angles: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),  # (roll, pitch, yaw)
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    print("euler_angles", euler_angles)
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = Rotation.from_euler("xyz", euler_angles, degrees=degrees).as_matrix()
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix


def interpolate_matrices(matrix1, matrix2, t):
    """Interpolate between two 4x4 matrices at time t [0, 1]."""
    q1, trans1 = decompose_matrix(matrix1)
    q2, trans2 = decompose_matrix(matrix2)

    # Linear interpolation for translation
    trans_interpolated = (1 - t) * trans1 + t * trans2

    # SLERP for rotation interpolation
    q_interpolated = Quaternion.slerp(q1, q2, t)
    rot_interpolated = q_interpolated.rotation_matrix

    # Recompose the interpolated matrix
    interpolated_matrix = recompose_matrix(rot_interpolated, trans_interpolated)
    return interpolated_matrix


def task_space_interpolation_goto(reachy_arm, target_pose) -> None:
    mat1 = reachy_arm.forward_kinematics()
    mat2 = target_pose
    # l2 distance between the two matrices in x, y, z only
    l2_distance_xyz = np.linalg.norm(mat1[:3, 3] - mat2[:3, 3])
    # distance in orientation TODO
    # speed = 0.1
    # nb_points = 20
    # duration = (l2_distance_xyz / speed) / nb_points
    freq = 120
    total_duration = 3.0
    nb_points = int(total_duration * freq)
    nb_points_final = int(1.0 * freq)
    try:
        l_ik_sol = reachy_arm.inverse_kinematics(mat2)
        goal_pose = reachy_arm.forward_kinematics(l_ik_sol)
        precision_distance_xyz_to_sol = np.linalg.norm(goal_pose[:3, 3] - mat2[:3, 3])
        print(f"l2 xyz distance Ik SOL vs goal pose: {precision_distance_xyz_to_sol}")
    except:
        print("Goal pose is not reachable!")
    for t in np.linspace(0, 1, nb_points):
        interpolated_matrix = interpolate_matrices(mat1, mat2, t)
        request = ArmCartesianGoal(
            id=reachy_arm._part_id,
            goal_pose=Matrix4x4(data=interpolated_matrix.flatten().tolist()),
        )
        reachy_arm._arm_stub.SendArmCartesianGoal(request)
        time.sleep(1 / freq)

    time.sleep(0.1)
    current_pose = reachy_arm.forward_kinematics()
    precision_distance_xyz = np.linalg.norm(current_pose[:3, 3] - mat2[:3, 3])
    if precision_distance_xyz > 0.003:
        print("Precision is not good enough, spamming the goal position!")
        for t in np.linspace(0, 1, nb_points):
            # Spamming the goal position to make sure its reached
            request = ArmCartesianGoal(
                id=reachy_arm._part_id,
                goal_pose=Matrix4x4(data=mat2.flatten().tolist()),
            )
            reachy_arm._arm_stub.SendArmCartesianGoal(request)
            time.sleep(1 / freq)

        time.sleep(0.1)
        current_pose = reachy_arm.forward_kinematics()
        precision_distance_xyz = np.linalg.norm(current_pose[:3, 3] - mat2[:3, 3])
    print(f"l2 xyz distance to goal: {precision_distance_xyz}")


def angle_diff(a: float, b: float) -> float:
    """Returns the smallest distance between 2 angles in rads"""
    d = a - b
    d = ((d + np.pi) % (2 * np.pi)) - np.pi
    return d


def get_ik(reachy: ReachySDK, control_ik: ControlIK, M: npt.NDArray[np.float64], arm: str, order_id: int) -> None:
    # control_ik = ControlIK()
    if arm == "r_arm":
        request = ArmCartesianGoal(
            id=reachy.r_arm._part_id,
            goal_pose=Matrix4x4(data=M.flatten().tolist()),
            # continuous_mode=IKContinuousMode.DISCRETE,
            constrained_mode=IKConstrainedMode.LOW_ELBOW,
            # constrained_mode=IKConstrainedMode.UNCONSTRAINED,
            preferred_theta=FloatValue(
                value=-5 * np.pi / 6,
            ),
            d_theta_max=FloatValue(value=0.1),
            order_id=Int32Value(value=order_id),
        )
        reachy.r_arm._arm_stub.SendArmCartesianGoal(request)
        # print(reachy.r_arm.shoulder.pitch.present_position)
    elif arm == "l_arm":
        request = ArmCartesianGoal(
            id=reachy.l_arm._part_id,
            goal_pose=Matrix4x4(data=M.flatten().tolist()),
            continuous_mode=IKContinuousMode.CONTINUOUS,
            constrained_mode=IKConstrainedMode.LOW_ELBOW,
            # constrained_mode=IKConstrainedMode.UNCONSTRAINED,
            preferred_theta=FloatValue(
                value=-3 * np.pi / 6,
            ),
            d_theta_max=FloatValue(value=0.1),
            order_id=Int32Value(value=(order_id + 1)),
        )
        reachy.l_arm._arm_stub.SendArmCartesianGoal(request)


def make_movement(reachy: ReachySDK) -> None:
    pose1 = [[0.38, -0.2, -0.28], [0.0, -np.pi/2, -np.pi/8]]
    pose2 = [[0.38, -0.2, -0.28], [0.0, -np.pi/4, -np.pi/8]]
    pose3 = [[0.38, -0.2, -0.28], [0.0, -np.pi/2, -np.pi/8]]
    pose4 = [[0.2, 0.3, -0.1], [0.0, -np.pi/4, np.pi/2]]
    pose5 = [[0.38, -0.2, -0.28], [0.0, -np.pi/2, -np.pi/8]]
    pose6 = [[0.5, -0.2, -0.15], [0.0, -np.pi/2, -np.pi/16]]
    pose7 = [[0.5, -0.2, -0.15], [0.0, -np.pi/4, -np.pi/16]]



    pose1_l = [[0.38, 0.2, -0.28], [0.0, -np.pi/2, np.pi/8]]
    pose2_l = [[0.38, 0.2, -0.28], [0.0, -np.pi/4, np.pi/8]]
    m1 = get_homogeneous_matrix_msg_from_euler(pose1[0], pose1[1])
    m2 = get_homogeneous_matrix_msg_from_euler(pose2[0], pose2[1])
    m3 = get_homogeneous_matrix_msg_from_euler(pose3[0], pose3[1])
    m4 = get_homogeneous_matrix_msg_from_euler(pose4[0], pose4[1])
    m5 = get_homogeneous_matrix_msg_from_euler(pose5[0], pose5[1])
    m6 = get_homogeneous_matrix_msg_from_euler(pose6[0], pose6[1])
    m7 = get_homogeneous_matrix_msg_from_euler(pose7[0], pose7[1])
    m1_l = get_homogeneous_matrix_msg_from_euler(pose1_l[0], pose1_l[1])
    m2_l = get_homogeneous_matrix_msg_from_euler(pose2_l[0], pose2_l[1])
    while True:
        task_space_interpolation_goto(reachy.r_arm, m1)
        task_space_interpolation_goto(reachy.r_arm, m2)
        task_space_interpolation_goto(reachy.r_arm, m3)
        # task_space_interpolation_goto(reachy.r_arm, m4)
        # task_space_interpolation_goto(reachy.r_arm, m5)
        task_space_interpolation_goto(reachy.r_arm, m6)
        task_space_interpolation_goto(reachy.r_arm, m7)

        # time.sleep(5)
        # task_space_interpolation_goto(reachy.l_arm, m1_l)
        # task_space_interpolation_goto(reachy.l_arm, m2_l)


def spam_pose(reachy: ReachySDK, control_ik: ControlIK, pose: npt.NDArray[np.float64]) -> None:
    order_id = 0
    start_time = time.time()
    l_pose = np.array([[pose[0][0], -pose[0][1], pose[0][2]], [-pose[1][0], pose[1][1], -pose[1][2]]])
    pose = get_homogeneous_matrix_msg_from_euler(pose[0], pose[1])
    l_pose = get_homogeneous_matrix_msg_from_euler(l_pose[0], l_pose[1])
    while time.time() - start_time < 5:
        get_ik(reachy, control_ik, pose, "r_arm", order_id)
        get_ik(reachy, control_ik, l_pose, "l_arm", order_id)
        order_id += 2
        time.sleep(0.01)


def main() -> None:
    control_ik = ControlIK()
    reachy = ReachySDK(host="localhost")

    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    # poses = [
    #     np.array([[0.001, -0.2, -0.659], [0.0, 0.0, 0.0]]),
    #     np.array([[0.38, -0.2, -0.28], [0.0, -np.pi / 2, 0.0]]),
    #     np.array([[0.30, -0.2, -0.28], [0.0, 0.0, np.pi / 3]]),  # top grasp
    #     np.array([[0.0, -0.2, -0.66], [0.0, 0.0, -np.pi / 3]]),  # backwards limit
    #     np.array([[0.001, -0.2, -0.68], [0.0, 0.0, -np.pi / 3]]),  # pose out of reach
    #     np.array([[0.001, -0.2, -0.659], [0.0, np.pi / 2, 0.0]]),  # wrist out of reach
    #     np.array([[0.38, -0.2, -0.28], [0.0, np.pi / 2, 0.0]]),  # wrist limit
    #     np.array([[0.1, -0.2, 0.0], [0.0, np.pi, 0.0]]),  # elbow limit
    #     np.array([[0.38, -0.2, -0.28], [0.0, 0.0, 0.0]]),  # shoulder limit?
    #     np.array([[0.1, 0.2, -0.1], [0.0, -np.pi / 2, np.pi / 2]]),  # shoulder limit
    # ]

    # for pose in poses:
    #     spam_pose(reachy, control_ik, pose)
    #     # time.sleep(5)

    make_movement(reachy)


if __name__ == "__main__":
    main()
