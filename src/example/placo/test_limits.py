from pathlib import Path

import numpy as np
import numpy.typing as npt
from reachy_placo.ik_reachy_placo import IKReachyQP

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import get_best_continuous_theta, tend_to_prefered_theta
from reachy2_symbolic_ik.utils_placo import go_to_position


def make_circle(
    symbolic_ik: SymbolicIK,
    placo_ik: IKReachyQP,
    prefered_theta: float,
    center: npt.NDArray[np.float64] = np.array([0.3, -0.4, -0.3]),
    radius: float = 0.1,
) -> None:
    if symbolic_ik is None:
        raise ValueError("symbolic_ik is None")
    orientation = np.array([0.0, -np.pi / 2, 0.0])
    Y = center[1] + radius * np.cos(np.linspace(0, 2 * np.pi, 100))
    Z = center[2] + radius * np.sin(np.linspace(0, 2 * np.pi, 100))
    X = center[0] * np.ones(100)
    previous_theta = prefered_theta
    while True:
        for i in range(100):
            goal_pose = [[X[i], Y[i], Z[i]], orientation]
            is_reachable, interval, get_joints = symbolic_ik.is_reachable(goal_pose)
            if is_reachable:
                is_reachable, theta = get_best_continuous_theta(
                    previous_theta, interval, get_joints, 0.05, prefered_theta, symbolic_ik.arm
                )
            else:
                print("Pose not reachable")
                is_reachable, interval, get_joints = symbolic_ik.is_reachable_no_limits(goal_pose)
                if is_reachable:
                    is_reachable, theta = tend_to_prefered_theta(
                        previous_theta, interval, get_joints, 0.05, goal_theta=prefered_theta
                    )
                else:
                    print("Pose not reachable________________")

            joints, elbow_position = get_joints(theta)
            previous_theta = theta
            go_to_position(placo_ik, joints, wait=0.0, arm=symbolic_ik.arm)


def singularity_test(symbolic_ik: SymbolicIK, placo_ik: IKReachyQP) -> None:
    position = np.array([0.0, -1000.0, 0.0])
    orientation = np.array([-np.pi / 2, 0.0, 0.0])
    goal_pose = np.array([position, orientation])
    is_reachable, interval, get_joints = symbolic_ik.is_reachable(goal_pose)
    if is_reachable:
        print("Pose reachable")
        print(interval)
        print(get_joints(interval[0]))
        print(get_joints(interval[1])[0])
        print(get_joints(0)[0])
        print(get_joints(2)[0])
        print(get_joints(-1.5)[0])
        go_to_position(placo_ik, get_joints(interval[0])[0], wait=3)
        joints2 = get_joints(interval[1])[0]
        print(joints2)
        # joints2[0] = joints2[0] + np.pi / 3
        # joints2[2] = joints2[2] - np.pi / 3

        joint = np.linspace(interval[0], interval[1], 50)
        while True:
            for i in range(50):
                joints2[0] = joint[i]
                joints2[2] = -2 * joint[i] - np.pi
                joints2[6] = -joint[i] - np.radians(-180)
                print(joints2)
                go_to_position(placo_ik, joints2, wait=0)
    else:
        print("Pose not reachable")
        is_reachable, interval, get_joints = symbolic_ik.is_reachable_no_limits(goal_pose)
        if is_reachable:
            print("Pose reachable")
            print(interval)
            print(get_joints(interval[0])[0])
            print(get_joints(interval[1])[0])
            print(get_joints(0)[0])
            print(get_joints(2)[0])
            print(get_joints(-1.5)[0])

            go_to_position(placo_ik, get_joints(interval[0])[0], wait=3)
            joints2 = get_joints(interval[1])[0]
            joints2[0] = joints2[0] + np.pi / 3
            joints2[2] = joints2[2] + np.pi / 3
            go_to_position(placo_ik, joints2, wait=8)

        else:
            print("Pose really not reachable")


def test_upperarm_singularity(symbolic_ik: SymbolicIK, placo_ik: IKReachyQP) -> None:
    goal_pose = [[0.38, -0.47, 0], [0, -np.pi / 2, 0]]
    is_reachable, interval, get_joints = symbolic_ik.is_reachable(goal_pose)
    if is_reachable:
        print("Pose reachable")
        print(interval)
        joints, elbow_position = get_joints(interval[0])
        theta0 = np.linspace(interval[0], np.pi, 50)
        theta1 = np.linspace(-np.pi, interval[1], 50)
        go_to_position(placo_ik, joints, wait=3)

        while True:
            for i in range(50):
                joints, elbow_position = get_joints(theta0[i])
                go_to_position(placo_ik, joints, wait=0)
            for i in range(50):
                joints, elbow_position = get_joints(theta1[i])
                go_to_position(placo_ik, joints, wait=0)

    else:
        print("Pose not reachable")
        is_reachable, interval, get_joints = symbolic_ik.is_reachable_no_limits(goal_pose)
        if is_reachable:
            print("Pose reachable")
            print(interval)
            joints, elbow_position = get_joints(interval[0])
            go_to_position(placo_ik, joints, wait=3)
        else:
            print("Pose really not reachable")


def test_arm_limits(symbolic_ik: SymbolicIK, placo_ik: IKReachyQP) -> None:
    goal_position = [00.05, -0.83, 0]
    goal_orientation = [-1.5, 0, 0]
    goal_pose = np.array([goal_position, goal_orientation])
    result = symbolic_ik.is_reachable(goal_pose)
    if result[0]:
        print("pose reachable")
        print(result[1])
        joints, elbow_position = result[2](result[1][0])
        go_to_position(placo_ik, joints, wait=3)
    else:
        print("Pose not reachable")
        result = symbolic_ik.is_reachable_no_limits(goal_pose)
        if result[0]:
            print("pose reachable")
            print(result[1])
            joints, elbow_position = result[2](result[1][0])
            go_to_position(placo_ik, joints, wait=3)
        else:
            print("Pose really not reachable")

    goal_position = [0.1, -0.74, 0]
    goal_orientation = [0, -1.57, -0.2]
    goal_pose = np.array([goal_position, goal_orientation])
    result = symbolic_ik.is_reachable(goal_pose)
    if result[0]:
        print("pose reachable")
        print(result[1])
        joints, elbow_position = result[2](result[1][0])
        go_to_position(placo_ik, joints, wait=3)
    else:
        print("Pose not reachable")
        result = symbolic_ik.is_reachable_no_limits(goal_pose)
        if result[0]:
            print("pose reachable")
            print(result[1])
            joints, elbow_position = result[2](result[1][0])
            go_to_position(placo_ik, joints, wait=3)
        else:
            print("Pose really not reachable")


def main_test() -> None:
    # symbolic_ik_r = SymbolicIK()
    symbolic_ik_r = SymbolicIK(
        # shoulder_orientation_offset=np.array([0.0, 0.0, 15]), shoulder_position=np.array([-0.0479, -0.1913, 0.025])
        shoulder_orientation_offset=np.array([0.0, 0.0, 0]),
        shoulder_position=np.array([0.0, -0.2, 0.0]),
    )
    # symbolib_ik_l = SymbolicIK(arm="l_arm")

    urdf_name = "reachy2_no_offset"
    # urdf_name = "reachy2_ik"

    urdf_path = Path("src/config_files")
    for file in urdf_path.glob("**/*.urdf"):
        if file.stem == urdf_name:
            urdf_path = file.resolve()
            break
    placo_ik = IKReachyQP(
        viewer_on=True,
        collision_avoidance=False,
        parts=["r_arm", "l_arm"],
        position_weight=1.9,
        orientation_weight=1e-2,
        robot_version="reachy_2",
        velocity_limit=50.0,
    )

    placo_ik.setup(urdf_path=str(urdf_path))
    placo_ik.create_tasks()

    test_arm_limits(symbolic_ik_r, placo_ik)
    # singularity_test(symbolic_ik_r, placo_ik)
    # test_upperarm_singularity(symbolic_ik_r, placo_ik)

    # goal_position = [0.10475819158237422, -0.10011326608169921, -0.6]
    # goal_orientation = [0, 0, 0]
    # goal_pose = np.array([goal_position, goal_orientation])

    # result_r = symbolic_ik_r.is_reachable(goal_pose)
    # if result_r[0]:
    #     print("pose reachable")
    #     print(result_r[1])
    #     joints, elbow_position = result_r[2](result_r[1][0])
    #     go_to_position(placo_ik, joints, wait=3)
    #     # is_reachable, theta = shoulder_limits(result_r[1], result_r[2])
    #     # print(theta)
    #     # if is_reachable:
    #     #     joints, elbow_position = result_r[2](result_r[1][0])
    #     #     go_to_position(placo_ik, joints, wait=3)
    #     #     is_correct = are_joints_correct(placo_ik, joints, goal_pose)
    #     #     print(is_correct)
    #     # else:
    #     #     print("Pose not reachable because of shoulder limits")

    # else:
    #     result_r = symbolic_ik_r.is_reachable_no_limits(goal_pose)
    #     if result_r[0]:
    #         print("pose reachable")
    #         print(result_r[1])
    #         joints, elbow_position = result_r[2](result_r[1][0])
    #         go_to_position(placo_ik, joints, wait=3)
    #         # is_reachable, theta = shoulder_limits(result_r[1], result_r[2])
    #         # print(theta)
    #         # if is_reachable:
    #         #     joints, elbow_position = result_r[2](result_r[1][0])
    #         #     go_to_position(placo_ik, joints, wait=3)
    #         #     is_correct = are_joints_correct(placo_ik, joints, goal_pose)
    #         #     print(is_correct)
    #         # else:
    #         #     print("Pose not reachable because of shoulder limits")
    #     else:
    #         print("Pose really not reachable")

    # # make_circle(symbolic_ik_r, placo_ik, 5 * np.pi / 4, np.array([0.3, -0.4, -0.3]), 0.4)


if __name__ == "__main__":
    main_test()
