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
            d_theta_max=FloatValue(value=0.05),
            order_id=Int32Value(value=5),
        )
        reachy.r_arm._stub.SendArmCartesianGoal(request)
        # print(reachy.r_arm.shoulder.pitch.present_position)
    elif arm == "l_arm":
        request = ArmCartesianGoal(
            id=reachy.l_arm._part_id,
            goal_pose=Matrix4x4(data=pose.flatten().tolist()),
            continuous_mode=IKContinuousMode.CONTINUOUS,
            constrained_mode=IKConstrainedMode.UNCONSTRAINED,
            preferred_theta=FloatValue(
                value=-4 * np.pi / 6,
            ),
            d_theta_max=FloatValue(value=0.05),
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
    # reachy.send_goal_positions()


def turn_wrist(reachy: ReachySDK, duration: float = 10.0) -> None:
    position = np.array([0.001, -0.2, -0.659])
    start_orientation = np.array([0.0, 0.0, 0.0])
    end_orientation = np.array([0.0, 0.0, 10])

    control_frequency = 100.0
    nbr_points = int(duration * control_frequency)

    for i in range(nbr_points):
        t = time.time()
        orientation = start_orientation + (end_orientation - start_orientation) * (i / nbr_points)
        rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
        pose = make_homogenous_matrix_from_rotation_matrix(position, rotation_matrix)
        go_to_pose(reachy, pose, "r_arm")

        # l_position = l_start_position + (l_end_position - l_start_position) * (i / nbr_points)
        # l_orientation = l_start_orientation + (l_end_orientation - l_start_orientation) * (i / nbr_points)
        # l_rotation_matrix = R.from_euler("xyz", l_orientation).as_matrix()
        # l_pose = make_homogenous_matrix_from_rotation_matrix(l_position, l_rotation_matrix)
        # go_to_pose(reachy, l_pose, "l_arm")

        time.sleep(max(1.0 / control_frequency - (time.time() - t), 0.0))


def turn_shoulder(reachy: ReachySDK, duration: float = 10.0):
    start_position = np.array([0.0, -0.2, -0.66])
    end_position = np.array([0.0, -0.86, 0.0])
    start_orientation = np.array([0.0, 0.0, 0.0])
    end_orientation = np.array([-np.pi/2, 0.0, 0.0])
    start_theta = 0.0
    end_theta = np.pi / 2

    control_frequency = 100.0
    nbr_points = int(duration * control_frequency)

    for i in range(nbr_points):
        t = time.time()
        theta = start_theta + (end_theta - start_theta) * (i / nbr_points)
        position = [0.0, -np.sin(theta) * 0.66 - 0.2,- np.cos(theta) * 0.66]
        orientation = start_orientation + (end_orientation - start_orientation) * (i / nbr_points)
        rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
        pose = make_homogenous_matrix_from_rotation_matrix(position, rotation_matrix)
        go_to_pose(reachy, pose, "r_arm")
        time.sleep(max(1.0 / control_frequency - (time.time() - t), 0.0))



if __name__ == "__main__":
    print("Trying to connect on localhost Reachy...")
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")


    reachy.turn_on()

    ik = [0.0, 15.0, -10.0, 0.0, 0.0, 0.0, 0.0]
    reachy.r_arm.goto(ik, 3.0, degrees=True, interpolation_mode="minimum_jerk")
    ik = [0.0, -15.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    reachy.l_arm.goto(ik, 3.0, degrees=True, interpolation_mode="minimum_jerk")
    time.sleep(3.0)
    turn_shoulder(reachy, 10.0)
    time.sleep(3.0)
    ik = [0.0, 15.0, -10.0, 0.0, 0.0, 0.0, 0.0]
    reachy.r_arm.goto(ik, 3.0, degrees=True, interpolation_mode="minimum_jerk")
    ik = [0.0, -15.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    reachy.l_arm.goto(ik, 3.0, degrees=True, interpolation_mode="minimum_jerk")
    time.sleep(3.0)
    # turn_wrist(reachy, 10.0)
