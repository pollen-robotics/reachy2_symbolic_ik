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


def main_test() -> None:
    symbolib_ik_r = SymbolicIK()
    # symbolib_ik_l = SymbolicIK(arm="l_arm")
    urdf_path = Path("src/config_files")
    for file in urdf_path.glob("**/*.urdf"):
        if file.stem == "reachy2_ik":
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

    goal_position = [0.9, -0.70, -0.20]
    goal_orientation = [0, -np.pi / 2, 0]
    goal_pose = np.array([goal_position, goal_orientation])

    result_r = symbolib_ik_r.is_reachable(goal_pose)
    if result_r[0]:
        print("pose reachable")
        print(result_r[1])
        joints, elbow_position = result_r[2](result_r[1][0])
        go_to_position(placo_ik, joints, wait=3)
        # is_reachable, theta = shoulder_limits(result_r[1], result_r[2])
        # print(theta)
        # if is_reachable:
        #     joints, elbow_position = result_r[2](result_r[1][0])
        #     go_to_position(placo_ik, joints, wait=3)
        #     is_correct = are_joints_correct(placo_ik, joints, goal_pose)
        #     print(is_correct)
        # else:
        #     print("Pose not reachable because of shoulder limits")

    else:
        result_r = symbolib_ik_r.is_reachable_no_limits(goal_pose)
        if result_r[0]:
            print("pose reachable")
            print(result_r[1])
            joints, elbow_position = result_r[2](result_r[1][0])
            go_to_position(placo_ik, joints, wait=3)
            # is_reachable, theta = shoulder_limits(result_r[1], result_r[2])
            # print(theta)
            # if is_reachable:
            #     joints, elbow_position = result_r[2](result_r[1][0])
            #     go_to_position(placo_ik, joints, wait=3)
            #     is_correct = are_joints_correct(placo_ik, joints, goal_pose)
            #     print(is_correct)
            # else:
            #     print("Pose not reachable because of shoulder limits")
        else:
            print("Pose really not reachable")

    make_circle(symbolib_ik_r, placo_ik, 5 * np.pi / 4, np.array([0.3, -0.4, -0.3]), 0.4)


if __name__ == "__main__":
    main_test()
