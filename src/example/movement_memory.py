from pathlib import Path

import numpy as np
import numpy.typing as npt
from reachy_placo.ik_reachy_placo import IKReachyQP

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import (
    angle_diff,
    get_best_continuous_theta,
    get_theta_from_current_pose,
)
from reachy2_symbolic_ik.utils_placo import go_to_position


def make_line(
    symbolic_ik: SymbolicIK,
    placo_ik: IKReachyQP,
    start_position: npt.NDArray[np.float64],
    end_position: npt.NDArray[np.float64],
    start_orientation: npt.NDArray[np.float64],
    end_orientation: npt.NDArray[np.float64],
    current_joints: npt.NDArray[np.float64],
    current_pose: npt.NDArray[np.float64],
    nb_points: int = 100,
) -> None:
    x = np.linspace(start_position[0], end_position[0], nb_points)
    y = np.linspace(start_position[1], end_position[1], nb_points)
    z = np.linspace(start_position[2], end_position[2], nb_points)
    roll = np.linspace(start_orientation[0], end_orientation[0], nb_points)
    pitch = np.linspace(start_orientation[1], end_orientation[1], nb_points)
    yaw = np.linspace(start_orientation[2], end_orientation[2], nb_points)

    is_reachable, intervalle, get_joints = symbolic_ik.is_reachable(current_pose)
    if is_reachable:
        is_reachable, theta0 = get_theta_from_current_pose(
            get_joints, intervalle, current_joints, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], symbolic_ik.arm
        )
        if not (is_reachable):
            if angle_diff(intervalle[0], intervalle[1]) > 0:
                print("OMG ANGLE DIFF > 0 ")
                theta0 = angle_diff(intervalle[0], intervalle[1]) / 2 + intervalle[1] + np.pi
            else:
                theta0 = angle_diff(intervalle[0], intervalle[1]) / 2 + intervalle[1]
    else:
        if angle_diff(intervalle[0], intervalle[1]) > 0:
            print("OMG ANGLE DIFF > 0 ")
            theta0 = angle_diff(intervalle[0], intervalle[1]) / 2 + intervalle[1] + np.pi
        else:
            theta0 = angle_diff(intervalle[0], intervalle[1]) / 2 + intervalle[1]

    for i in range(nb_points):
        goal_pose = [[x[i], y[i], z[i]], [roll[i], pitch[i], yaw[i]]]
        result = symbolic_ik.is_reachable(goal_pose)
        is_reachable, theta = get_best_continuous_theta(theta0, intervalle, get_joints, 0.1, symbolic_ik.arm)
        if is_reachable:
            joints, elbow_position = result[2](theta)
            go_to_position(placo_ik, joints, wait=0.0, arm=symbolic_ik.arm)
        else:
            print("Pose not reachable")


def make_square(
    symbolic_ik: SymbolicIK,
    placo_ik: IKReachyQP,
    current_joints: npt.NDArray[np.float64],
    current_pose: npt.NDArray[np.float64],
) -> None:
    if symbolic_ik is None:
        raise ValueError("symbolic_ik is None")
    orientation = np.array([0.0, -np.pi / 2, 0.0])
    start_positions = []
    end_positions = []
    if symbolic_ik.arm == "r_arm":
        print("r_arm")
        start_positions.append(np.array([0.4, -0.5, -0.3]))
        end_positions.append(np.array([0.4, -0.5, -0.0]))
        start_positions.append(np.array([0.4, -0.5, -0.0]))
        end_positions.append(np.array([0.4, -0.3, -0.0]))
        start_positions.append(np.array([0.4, -0.3, -0.0]))
        end_positions.append(np.array([0.4, -0.3, -0.3]))
        start_positions.append(np.array([0.4, -0.3, -0.3]))
        end_positions.append(np.array([0.4, -0.5, -0.3]))
    else:
        print("l_arm")
        start_positions.append(np.array([0.4, 0.5, -0.3]))
        end_positions.append(np.array([0.4, 0.5, -0.0]))
        start_positions.append(np.array([0.4, 0.5, -0.0]))
        end_positions.append(np.array([0.4, 0.3, -0.0]))
        start_positions.append(np.array([0.4, 0.3, -0.0]))
        end_positions.append(np.array([0.4, 0.3, -0.3]))
        start_positions.append(np.array([0.4, 0.3, -0.3]))
        end_positions.append(np.array([0.4, 0.5, -0.3]))

    while True:
        for i in range(len(start_positions)):
            make_line(
                symbolic_ik,
                placo_ik,
                start_positions[i],
                end_positions[i],
                orientation,
                orientation,
                current_joints,
                current_pose,
                nb_points=30,
            )

            # current_joints = [
            #     placo_ik.robot.r_arm.r_shoulder_pitch,
            #     # reachy.r_arm.r_shoulder_pitch,
            #     # reachy.r_arm.r_shoulder_roll,
            #     # reachy.r_arm.r_arm_yaw,
            #     # reachy.r_arm.r_elbow_pitch,
            #     # reachy.r_arm.r_forearm_yaw,
            #     # reachy.r_arm.r_wrist_pitch,
            #     # reachy.r_arm.r_wrist_roll,
            # ]


def main_test() -> None:
    symbolic_ik_r = SymbolicIK()
    # symbolic_ik_l = SymbolicIK(arm="l_arm")
    urdf_path = Path("src/config_files")
    for file in urdf_path.glob("**/*.urdf"):
        if file.stem == "reachy2_ik":
            urdf_path = file.resolve()
            break
    placo_ik = IKReachyQP(
        viewer_on=True,
        collision_avoidance=False,
        parts=["r_arm"],
        position_weight=1.9,
        orientation_weight=1e-2,
        robot_version="reachy_2",
        velocity_limit=50.0,
    )
    placo_ik.setup(urdf_path=str(urdf_path))
    placo_ik.create_tasks()

    # goal_position = [0.20, -0.2, -0.0]
    # goal_orientation = np.array([-20, -60, 10])
    # goal_orientation = np.deg2rad(goal_orientation)
    # goal_pose = [goal_position, goal_orientation]
    # make_movement_test(symbolic_ik_r, placo_ik, goal_pose)

    # make_square(symbolic_ik_l, placo_ik)

    # start_position = np.array([0.4, 0.1, -0.4])
    # end_position = np.array([0.3, -0.2, -0.1])
    # start_orientation = np.array([0.35, -1.40, 0.17])
    # end_orientation = np.array([0.0, -0.0, 0.0])
    # make_line(symbolic_ik_r, placo_ik, start_position, end_position, start_orientation, end_orientation, nb_points=300)

    current_pose = np.array([[0.4, -0.5, -0.3], [0.0, -np.pi / 2, 0.0]])
    result = symbolic_ik_r.is_reachable(current_pose)
    if result[0]:
        joints, elbow_position = result[2](result[1][0])
        make_square(symbolic_ik_r, placo_ik, joints, current_pose)

    #     while True:
    #         start_position = np.array([0.4, -0.5, -0.3])
    #         end_position = np.array([0.4, -0.5, -0.0])
    #         start_orientation = np.array([0.0, -np.pi / 2, 0.0])
    #         end_orientation = np.array([0.0, -np.pi / 2, 0.0])
    #         make_line(
    #             symbolic_ik_r,
    #             placo_ik,
    #             start_position,
    #             end_position,
    #             start_orientation,
    #             end_orientation,
    #             joints,
    #             current_pose,
    #             nb_points=50,
    #         )
    else:
        print("Pose not reachable")


if __name__ == "__main__":
    main_test()
