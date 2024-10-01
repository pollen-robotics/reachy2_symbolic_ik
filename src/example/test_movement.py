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
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.utils import make_homogenous_matrix_from_rotation_matrix

# TODO fix this test : movement test can not work with the discrete control function


def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> None:
    if arm == "r_arm":
        request = ArmCartesianGoal(
            id=reachy.r_arm._part_id,
            goal_pose=Matrix4x4(data=pose.flatten().tolist()),
            continuous_mode=IKContinuousMode.CONTINUOUS,
            constrained_mode=IKConstrainedMode.UNCONSTRAINED,
            preferred_theta=FloatValue(
                value=-4 * np.pi / 6,
            ),
            d_theta_max=FloatValue(value=0.01),
            order_id=Int32Value(value=5),
        )
        reachy.r_arm._stub.SendArmCartesianGoal(request)
        # print(reachy.r_arm.shoulder.pitch.present_position)
    elif arm == "l_arm":
        request = ArmCartesianGoal(
            id=reachy.l_arm._part_id,
            goal_pose=Matrix4x4(data=pose.flatten().tolist()),
            continuous_mode=IKContinuousMode.CONTINUOUS,
            constrained_mode=IKConstrainedMode.LOW_ELBOW,
            preferred_theta=FloatValue(
                value=-4 * np.pi / 6,
            ),
            d_theta_max=FloatValue(value=0.01),
            order_id=Int32Value(value=5),
        )
        reachy.l_arm._stub.SendArmCartesianGoal(request)

    # if arm == "r_arm":
    #     ik = reachy.r_arm.inverse_kinematics(pose)
    #     # real_pose = reachy.r_arm.forward_kinematics(ik)
    #     # pose_diff = np.linalg.norm(pose - real_pose)
    #     # print(f"pose diff {pose_diff}")
    #     for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik):
    #         joint.goal_position = goal_pos
    # elif arm == "l_arm":
    #     ik = reachy.l_arm.inverse_kinematics(pose)
    #     # real_pose = reachy.l_arm.forward_kinematics(ik)
    #     # pose_diff = np.linalg.norm(pose - real_pose)
    #     # print(f"pose diff {pose_diff}")
    #     for joint, goal_pos in zip(reachy.l_arm.joints.values(), ik):
    #         joint.goal_position = goal_pos
    reachy.send_goal_positions()


def make_line(
    reachy: ReachySDK, start_pose: npt.NDArray[np.float64], end_pose: npt.NDArray[np.float64], nbr_points: int = 100
) -> None:
    start_position = start_pose[0]
    end_position = end_pose[0]
    start_orientation = start_pose[1]
    end_orientation = end_pose[1]

    # Left arm
    l_start_position = np.array([start_position[0], -start_position[1], start_position[2]])
    l_end_position = np.array([end_position[0], -end_position[1], end_position[2]])
    l_start_orientation = np.array([-start_orientation[0], start_orientation[1], -start_orientation[2]])
    l_end_orientation = np.array([-end_orientation[0], end_orientation[1], -end_orientation[2]])

    for i in range(nbr_points):
        position = start_position + (end_position - start_position) * (i / nbr_points)
        orientation = start_orientation + (end_orientation - start_orientation) * (i / nbr_points)
        rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
        pose = make_homogenous_matrix_from_rotation_matrix(position, rotation_matrix)
        go_to_pose(reachy, pose, "r_arm")

        l_position = l_start_position + (l_end_position - l_start_position) * (i / nbr_points)
        l_orientation = l_start_orientation + (l_end_orientation - l_start_orientation) * (i / nbr_points)
        l_rotation_matrix = R.from_euler("xyz", l_orientation).as_matrix()
        l_pose = make_homogenous_matrix_from_rotation_matrix(l_position, l_rotation_matrix)
        go_to_pose(reachy, l_pose, "l_arm")

        time.sleep(0.05)


def make_circle(
    reachy: ReachySDK,
    center: npt.NDArray[np.float64],
    orientation: npt.NDArray[np.float64],
    radius: float,
    nbr_points: int = 100,
    number_of_turns: int = 3,
) -> None:
    Y_r = center[1] + radius * np.cos(np.linspace(0, 2 * np.pi, nbr_points))
    Z = center[2] + radius * np.sin(np.linspace(0, 2 * np.pi, nbr_points))
    X = center[0] * np.ones(nbr_points)
    Y_l = -center[1] + radius * np.cos(np.linspace(0, 2 * np.pi, nbr_points))

    for k in range(number_of_turns):
        for i in range(nbr_points):
            position = np.array([X[i], Y_r[i], Z[i]])
            rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
            pose = make_homogenous_matrix_from_rotation_matrix(position, rotation_matrix)
            go_to_pose(reachy, pose, "r_arm")

            l_position = np.array([X[i], Y_l[i], Z[i]])
            l_rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
            l_pose = make_homogenous_matrix_from_rotation_matrix(l_position, l_rotation_matrix)
            go_to_pose(reachy, l_pose, "l_arm")

            time.sleep(0.01)


def make_rectangle(
    reachy: ReachySDK,
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
    D: npt.NDArray[np.float64],
    nbr_points: int = 20,
    number_of_turns: int = 3,
) -> None:
    orientation = [0, -np.pi / 2, 0]

    for i in range(number_of_turns):
        make_line(reachy, np.array([A, orientation]), np.array([B, orientation]), nbr_points)
        make_line(reachy, np.array([B, orientation]), np.array([C, orientation]), nbr_points)
        make_line(reachy, np.array([C, orientation]), np.array([D, orientation]), nbr_points)
        make_line(reachy, np.array([D, orientation]), np.array([A, orientation]), nbr_points)


def turn_hand(reachy: ReachySDK, position: npt.NDArray[np.float64], orientation_init: npt.NDArray[np.float64]) -> None:
    orientation = [orientation_init[0], orientation_init[1], orientation_init[2]]
    for j in range(2):
        for i in range(100):
            rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
            pose = make_homogenous_matrix_from_rotation_matrix(position, rotation_matrix)
            print(pose)
            go_to_pose(reachy, pose, "r_arm")
            time.sleep(0.05)
            orientation[2] += 2 * np.pi / 100
            if orientation[2] > np.pi:
                orientation[2] = -np.pi
            print(orientation)


def main_test() -> None:
    print("Trying to connect on localhost Reachy...")
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    print("Making a line")
    start_pose = np.array([[0.38, -0.2, -0.28], [0, -np.pi / 2, 0]])
    end_pose = np.array([[0.38, -0.2, 0.28], [0, -np.pi, 0]])
    make_line(reachy, start_pose, end_pose)
    start_pose = np.array([[0.38, -0.2, 0.28], [0, -np.pi, 0]])
    end_pose = np.array([[0.0001, -0.2, 0.6599], [0, -np.pi, 0]])
    make_line(reachy, start_pose, end_pose)
    start_pose = np.array([[0.0001, -0.2, 0.6599], [0, -np.pi, 0]])
    end_pose = np.array([[0.0001, -0.859, 0.0], [-np.pi / 2, 0, 0]])
    make_line(reachy, start_pose, end_pose)
    start_pose = np.array([[0.0001, -0.859, 0.0], [-np.pi / 2, 0, 0]])
    end_pose = np.array([[0.38, -0.2, -0.28], [0, -np.pi / 2, 0]])
    make_line(reachy, start_pose, end_pose)

    start_pose = np.array([[0.38, -0.2, -0.28], [0, -np.pi / 2, 0]])
    end_pose = np.array([[0.15, -0.4, -0.30], [0, 0, np.pi / 2]])
    make_line(reachy, start_pose, end_pose)
    start_pose = np.array([[0.15, -0.4, -0.30], [0, 0, np.pi / 2]])
    end_pose = np.array([[0.15, -0.4, -0.25], [0, 0, np.pi / 2]])
    make_line(reachy, start_pose, end_pose)
    # start_pose = np.array([[0.18, -0.4, -0.30], [0, 0, np.pi / 2]])
    # end_pose = np.array([[0.08, -0.4, -0.30], [0, 0, np.pi / 2]])
    # make_line(reachy, start_pose, end_pose)
    # start_pose = np.array([[0.08, -0.4, -0.30], [0, 0, np.pi / 2]])
    # end_pose = np.array([[0.18, -0.4, -0.30], [0, 0, np.pi / 2]])
    # make_line(reachy, start_pose, end_pose)
    start_pose = np.array([[0.15, -0.4, -0.25], [0, 0, np.pi / 2]])
    end_pose = np.array([[0.15, -0.4, -0.20], [0, 0, np.pi / 2]])
    make_line(reachy, start_pose, end_pose)
    start_pose = np.array([[0.15, -0.4, -0.20], [0, 0, np.pi / 2]])
    end_pose = np.array([[0.15, -0.4, -0.20], [0, 0, np.pi / 2]])
    make_line(reachy, start_pose, end_pose)
    start_pose = np.array([[0.15, -0.4, -0.20], [0, 0, np.pi / 2]])
    end_pose = np.array([[0.38, -0.2, -0.28], [0, -np.pi / 2, 0]])
    make_line(reachy, start_pose, end_pose)
    start_pose = np.array([[0.38, -0.2, -0.28], [0, -np.pi / 2, 0]])
    end_pose = np.array([[0.38, -0.2, -0.28], [0, -np.pi / 2, 0]])
    make_line(reachy, start_pose, end_pose)

    # start_pose = np.array([[0.18, -0.4, -0.30], [0, 0, np.pi / 2]])
    # end_pose = np.array([[0.38, -0.2, -0.28], [0, -np.pi / 2, 0]])
    # make_line(reachy, start_pose, end_pose)
    # start_pose = np.array([[0.38, -0.2, -0.28], [0, -np.pi / 2, 0]])

    time.sleep(10.0)

    # print("Turning the hand")
    # position = np.array([0.001, -0.2, -0.6599])
    # orientation = np.array([0, 0, 0])
    # turn_hand(reachy, position, orientation)

    # time.sleep(1.0)

    print("Making a circle")
    center = np.array([0.4, -0.4, -0.2])
    orientation = np.array([0, -np.pi / 2, 0])
    radius = 0.15
    make_circle(reachy, center, orientation, radius)

    time.sleep(1.0)

    print("Making a rectangle")
    A = np.array([0.4, -0.5, -0.3])
    B = np.array([0.4, -0.5, -0.1])
    C = np.array([0.4, -0.2, -0.1])
    D = np.array([0.4, -0.2, -0.3])
    make_rectangle(reachy, A, B, C, D)

    time.sleep(1.0)

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


if __name__ == "__main__":
    main_test()
