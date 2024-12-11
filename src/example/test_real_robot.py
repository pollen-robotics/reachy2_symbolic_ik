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

# TODO fix poses to have reachable poses


def go_to_pose(  # noqa: C901
    reachy: ReachySDK,
    pose: npt.NDArray[np.float64],
    arm: str,
    duration: float = 4.0,
    goto_mode: str = "joint",
    interpolation_mode: str = "minimum_jerk",
    wait: bool = False,
    arc_direction: str = "",
    elliptic_radius: float = 0.1,
) -> None:
    if arm == "l_arm":
        pose = np.array([[pose[0][0], -pose[0][1], pose[0][2]], [-pose[1][0], pose[1][1], -pose[1][2]]])
    pose = make_homogenous_matrix_from_rotation_matrix(pose[0], R.from_euler("xyz", pose[1]).as_matrix())
    # print(pose)
    if goto_mode == "joint":
        if arm == "r_arm":
            ik = reachy.r_arm.inverse_kinematics(pose)
            reachy.r_arm.goto(ik, duration, degrees=True, interpolation_mode=interpolation_mode, wait=wait)
        elif arm == "l_arm":
            ik = reachy.l_arm.inverse_kinematics(pose)
            reachy.l_arm.goto(ik, duration, degrees=True, interpolation_mode=interpolation_mode, wait=wait)

    elif goto_mode == "matrix":
        if arm == "r_arm":
            reachy.r_arm.goto(pose, duration, interpolation_mode=interpolation_mode, wait=wait)
        elif arm == "l_arm":
            reachy.l_arm.goto(pose, duration, interpolation_mode=interpolation_mode, wait=wait)

    elif goto_mode == "teleop":
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

    elif goto_mode == "linear_cartesian":
        if arm == "r_arm":
            reachy.r_arm.send_cartesian_interpolation(pose, duration)
        elif arm == "l_arm":
            reachy.l_arm.send_cartesian_interpolation(pose, duration)

    elif goto_mode == "circular_cartesian":
        if arm == "r_arm":
            reachy.r_arm.send_cartesian_interpolation(pose, duration, arc_direction, elliptic_radius)
        elif arm == "l_arm":
            reachy.l_arm.send_cartesian_interpolation(pose, duration, arc_direction, elliptic_radius)


def make_line(
    reachy: ReachySDK, start_pose: npt.NDArray[np.float64], end_pose: npt.NDArray[np.float64], duration: float = 5.0
) -> None:
    if len(start_pose) == 0:
        start_pose = reachy.r_arm.forward_kinematics()
        start_pose = np.array([start_pose[:3, 3], R.from_matrix(start_pose[:3, :3]).as_euler("xyz")])
        # TODO : fix l_arm
        # l_start_pose = reachy.l_arm.forward_kinematics()
        # l_start_pose = [l_start_pose[:3, 3], R.from_matrix(l_start_pose[:3, :3]).as_euler("xyz")]
        # print(f"Start pose: {start_pose}")
        # print(f"l_Start pose: {l_start_pose}")
    start_position = start_pose[0]
    end_position = end_pose[0]
    start_orientation = start_pose[1]
    end_orientation = end_pose[1]

    control_frequency = 100.0
    nbr_points = int(duration * control_frequency)
    # Left arm
    # l_start_position = np.array([start_position[0], -start_position[1], start_position[2]])
    # l_end_position = np.array([end_position[0], -end_position[1], end_position[2]])
    # l_start_orientation = np.array([-start_orientation[0], start_orientation[1], -start_orientation[2]])
    # l_end_orientation = np.array([-end_orientation[0], end_orientation[1], -end_orientation[2]])

    for i in range(nbr_points):
        t = time.time()
        position = start_position + (end_position - start_position) * (i / nbr_points)
        orientation = start_orientation + (end_orientation - start_orientation) * (i / nbr_points)
        pose = np.array([position, orientation])
        go_to_pose(reachy, pose, "r_arm", duration, "teleop")

        # l_position = l_start_position + (l_end_position - l_start_position) * (i / nbr_points)
        # l_orientation = l_start_orientation + (l_end_orientation - l_start_orientation) * (i / nbr_points)
        pose = np.array([position, orientation])

        go_to_pose(reachy, pose, "l_arm", duration, "teleop")

        time.sleep(max(1.0 / control_frequency - (time.time() - t), 0.0))


def make_movement(
    reachy: ReachySDK, poses: npt.NDArray[np.float64], duration: float, goto_mode: str, interpolation_mode: str = ""
) -> None:
    if goto_mode == "matrix" or goto_mode == "joint":
        for pose in poses:
            go_to_pose(reachy, pose, "r_arm", duration, goto_mode, interpolation_mode, False)
            go_to_pose(reachy, pose, "l_arm", duration, goto_mode, interpolation_mode, True)

    elif goto_mode == "teleop":
        make_line(reachy, np.array([]), poses[0], duration)
        for i in range(len(poses) - 1):
            make_line(reachy, poses[i], poses[i + 1], duration)

    elif goto_mode == "linear_cartesian":
        for pose in poses:
            go_to_pose(reachy, pose, "r_arm", duration, goto_mode)
            time.sleep(0.5)


def square_test(
    reachy: ReachySDK, goto_mode: str, interpolation_mode: str = "", number_of_turns: int = 2, duration: float = 3.0
) -> None:
    A = np.array([[0.4, -0.5, -0.2], [0.0, -np.pi / 2, 0.0]])
    B = np.array([[0.4, -0.3, -0.2], [0.0, -np.pi / 2, 0.0]])
    C = np.array([[0.4, -0.3, 0], [0.0, -np.pi / 2, 0.0]])
    D = np.array([[0.4, -0.5, 0], [0.0, -np.pi / 2, 0.0]])
    if goto_mode == "circular_cartesian":
        go_to_pose(reachy, A, "r_arm", duration, "linear_cartesian")
        time.sleep(0.5)
        for _ in range(number_of_turns):
            go_to_pose(reachy, B, "r_arm", duration, "circular_cartesian", arc_direction="below", elliptic_radius=0.1)
            time.sleep(0.5)
            go_to_pose(reachy, C, "r_arm", duration, "circular_cartesian", arc_direction="left", elliptic_radius=0.1)
            time.sleep(0.5)
            go_to_pose(reachy, D, "r_arm", duration, "circular_cartesian", arc_direction="above", elliptic_radius=0.1)
            time.sleep(0.5)
            go_to_pose(reachy, A, "r_arm", duration, "circular_cartesian", arc_direction="right", elliptic_radius=0.1)
            time.sleep(0.5)
    else:
        for _ in range(number_of_turns):
            poses = np.array([A, B, C, D])
            make_movement(reachy, poses, duration, goto_mode, interpolation_mode)


def mobile_base_test(reachy: ReachySDK) -> None:
    if reachy.mobile_base is None:
        print("No mobile base found")
    else:
        reachy.mobile_base.translate_by(0.30, 0.0)
        time.sleep(2)
        reachy.mobile_base.translate_by(-0.30, 0.0)
        time.sleep(2)
        reachy.mobile_base.translate_by(0.0, 0.30)
        time.sleep(2)
        reachy.mobile_base.translate_by(0.0, -0.30)
        time.sleep(2)
        reachy.mobile_base.rotate_by(np.pi / 2)
        time.sleep(2)
        reachy.mobile_base.rotate_by(-np.pi / 2)
        time.sleep(2)


def test_ik(reachy: ReachySDK) -> None:
    poses = [
        np.array([[0.38, -0.2, -0.28], [0.0, -np.pi / 2, 0.0]]),
        np.array([[0.20, -0.025, -0.28], [0.0, -np.pi / 2, np.pi / 4]]),
        np.array([[0.38, -0.2, -0.28], [0.0, -np.pi / 2, 0.0]]),
        np.array([[0.659, -0.2, -0.0], [0.0, -np.pi / 2, 0.0]]),
        np.array([[0.001, -0.2, -0.659], [0.0, 0.0, 0.0]]),
    ]

    for arm in ["r_arm", "l_arm"]:
        for pose in poses:
            go_to_pose(reachy, pose, arm, duration=3.0, goto_mode="joint", interpolation_mode="minimum_jerk", wait=True)
            time.sleep(3)

    poses = [
        np.array([[0.38, -0.2, -0.28], [0.0, -np.pi / 2, 0.0]]),
        np.array([[0.2, -0.15, -0.2], [0.0, -np.pi / 2, np.pi / 2]]),
        np.array([[0.2, -0.058, -0.2], [0.0, -np.pi / 2, np.pi / 2]]),
        np.array([[0.001, -0.2, -0.659], [0.0, 0.0, 0.0]]),
    ]

    for pose in poses:
        go_to_pose(reachy, pose, "r_arm", duration=4.0, goto_mode="matrix", interpolation_mode="minimum_jerk", wait=False)
        go_to_pose(reachy, pose, "l_arm", duration=4.0, goto_mode="matrix", interpolation_mode="minimum_jerk", wait=True)
        time.sleep(5)


def go_to_zero(reachy: ReachySDK) -> None:
    ik = [0.0, 15.0, -10.0, 0.0, 0.0, 0.0, 0.0]
    reachy.r_arm.goto(ik, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=False)
    ik = [0.0, -15.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    reachy.l_arm.goto(ik, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=True)


def test_gripper(reachy: ReachySDK) -> None:
    reachy.r_arm.gripper.close()
    reachy.l_arm.gripper.close()
    time.sleep(1)
    reachy.r_arm.gripper.open()
    reachy.l_arm.gripper.open()
    time.sleep(1)
    reachy.r_arm.gripper.close()
    reachy.l_arm.gripper.close()
    time.sleep(1)


def movement_limit_test(reachy: ReachySDK, duration: float, goto_mode: str, interpolation_mode: str = "") -> None:
    poses = np.array(
        [
            np.array([[0.0001, -0.2, -0.6599], [0, 0, 0]]),
            np.array([[0.38, -0.2, -0.28], [0.0, -np.pi / 2, 0.0]]),
            np.array([[0.38, -0.2, 0.28], [0.0, -np.pi, 0.0]]),
            np.array([[0.0001, -0.2, 0.6599], [0.0, -np.pi, 0.0]]),
            np.array([[0.0001, -0.859, 0.0], [-np.pi / 2, 0, 0]]),
            np.array([[0.38, -0.2, -0.28], [0.0, -np.pi / 2, 0.0]]),
            np.array([[0.15, -0.4, -0.30], [0.0, 0.0, np.pi / 2]]),
            np.array([[0.15, -0.4, -0.25], [0.0, 0.0, np.pi / 2]]),
            np.array([[0.15, -0.4, -0.20], [0.0, 0.0, np.pi / 2]]),
            np.array([[0.38, -0.2, -0.28], [0.0, -np.pi / 2, 0.0]]),
            np.array([[0.0001, -0.2, -0.6599], [0.0, 0.0, 0.0]]),
        ]
    )

    make_movement(reachy, poses, duration, goto_mode, interpolation_mode)


def full_test() -> None:
    reachy = ReachySDK(host="localhost")

    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    try:
        go_to_zero(reachy)

        print("_____ Gripper test _____")
        test_gripper(reachy)

        print("_____ IK test _____")
        test_ik(reachy)
        go_to_zero(reachy)
        time.sleep(3)

        print("_____ Square test _____")
        print("Joint, minimum jerk")
        square_test(reachy, "joint", "minimum_jerk")
        print("Joint, linear")
        square_test(reachy, "joint", "linear")
        print("Matrix, minimum jerk")
        square_test(reachy, "matrix", "minimum_jerk")
        print("Matrix, linear")
        square_test(reachy, "matrix", "linear")
        print("Fake teleop")
        square_test(reachy, "teleop")
        print("Linear cartesian")
        square_test(reachy, "linear_cartesian")
        print("Circular cartesian")
        square_test(reachy, "circular_cartesian")

        print("_____ Movement limit test _____")
        print("Fake teleop")
        movement_limit_test(reachy, 4.0, "teleop")

        print("_____ Mobile base test _____")
        mobile_base_test(reachy)

        go_to_zero(reachy)
        time.sleep(3)

    except Exception as e:
        print(f"An error occurred: {e}")
        # print traceback
        import traceback

        traceback.print_exc()

    finally:
        reachy.cancel_all_goto()
        reachy.goto_posture("default", wait=True)
        # wait_for_pose_to_finish(goto_ids)
        reachy.r_arm.gripper.open()
        reachy.l_arm.gripper.open()

        print("Turning off Reachy")
        reachy.turn_off()

        time.sleep(0.2)

        exit("Exiting example")


def main() -> None:
    full_test()


if __name__ == "__main__":
    main()
