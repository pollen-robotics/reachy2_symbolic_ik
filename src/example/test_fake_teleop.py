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

    poses = [
        np.array([[0.001, -0.2, -0.659], [0.0, 0.0, 0.0]]),
        np.array([[0.38, -0.2, -0.28], [0.0, -np.pi / 2, 0.0]]),
        np.array([[0.30, -0.2, -0.28], [0.0, 0.0, np.pi / 3]]),  # top grasp
        np.array([[0.0, -0.2, -0.66], [0.0, 0.0, -np.pi / 3]]),  # backwards limit
        np.array([[0.001, -0.2, -0.68], [0.0, 0.0, -np.pi / 3]]),  # pose out of reach
        np.array([[0.001, -0.2, -0.659], [0.0, np.pi / 2, 0.0]]),  # wrist out of reach
        np.array([[0.38, -0.2, -0.28], [0.0, np.pi / 2, 0.0]]),  # wrist limit
        np.array([[0.1, -0.2, 0.0], [0.0, np.pi, 0.0]]),  # elbow limit
        np.array([[0.38, -0.2, -0.28], [0.0, 0.0, 0.0]]),  # shoulder limit?
        np.array([[0.1, 0.2, -0.1], [0.0, -np.pi / 2, np.pi / 2]]),  # shoulder limit
    ]

    for pose in poses:
        spam_pose(reachy, control_ik, pose)
        # time.sleep(5)


if __name__ == "__main__":
    main()
