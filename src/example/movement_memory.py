import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
from reachy_placo.ik_reachy_placo import IKReachyQP
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import get_best_continuous_theta, tend_to_prefered_theta
from reachy2_symbolic_ik.utils_placo import go_to_position


def make_line(
    symbolic_ik: SymbolicIK,
    placo_ik: IKReachyQP,
    start_position: npt.NDArray[np.float64],
    end_position: npt.NDArray[np.float64],
    start_orientation: npt.NDArray[np.float64],
    end_orientation: npt.NDArray[np.float64],
    prefered_theta: float,
    previous_theta: float = np.pi,
    nb_points: int = 100,
    init: bool = False,
) -> float:
    x = np.linspace(start_position[0], end_position[0], nb_points)
    y = np.linspace(start_position[1], end_position[1], nb_points)
    z = np.linspace(start_position[2], end_position[2], nb_points)
    roll = np.linspace(start_orientation[0], end_orientation[0], nb_points)
    pitch = np.linspace(start_orientation[1], end_orientation[1], nb_points)
    yaw = np.linspace(start_orientation[2], end_orientation[2], nb_points)
    # previous_theta = theta0

    for i in range(nb_points * 2):
        if init:
            if i < nb_points:
                i = 0
            else:
                i = i - nb_points
        else:
            if i >= nb_points:
                break
        goal_pose = [[x[i], y[i], z[i]], [roll[i], pitch[i], yaw[i]]]
        is_reachable, interval, get_joints = symbolic_ik.is_reachable(goal_pose)
        if is_reachable:
            is_reachable, theta = get_best_continuous_theta(
                previous_theta,
                interval,
                get_joints,
                0.05,
                prefered_theta,
                symbolic_ik.arm,
            )
            print(theta)
            previous_theta = theta

        else:
            print("Pose not reachable")
            is_reachable, interval, get_joints = symbolic_ik.is_reachable_no_limits(goal_pose)
            if is_reachable:
                is_reachable, theta = tend_to_prefered_theta(
                    previous_theta, interval, get_joints, 0.05, goal_theta=prefered_theta
                )
                previous_theta = theta
            else:
                print("Pose not reachable________________")

        joints, elbow_position = get_joints(theta)
        go_to_position(placo_ik, joints, wait=0.0, arm=symbolic_ik.arm)

    return float(theta)


def random_movement(symbolic_ik: SymbolicIK, placo_ik: IKReachyQP, prefered_theta: float) -> None:
    previous_position = np.array([0.4, -0.5, -0.3])
    orientation = np.array([0.0, -np.pi / 2, 0.0])
    previous_theta = prefered_theta
    while True:
        goal_x = np.random.uniform(0.2, 0.6)
        goal_y = np.random.uniform(-0.6, 0)
        goal_z = np.random.uniform(-0.4, 0.4)
        goal_position = np.array([goal_x, goal_y, goal_z])
        nb_points = np.linalg.norm(goal_position - previous_position) * 150
        previous_theta = make_line(
            symbolic_ik,
            placo_ik,
            previous_position,
            goal_position,
            orientation,
            orientation,
            prefered_theta,
            previous_theta,
            int(nb_points),
        )
        previous_position = goal_position


def make_circle(
    symbolic_ik: SymbolicIK,
    placo_ik: IKReachyQP,
    prefered_theta: float,
    center: npt.NDArray[np.float64] = np.array([0.3, -0.4, -0.3]),
    radius: float = 0.1,
    top: bool = False,
    moving_orientation: bool = False,
) -> None:
    if symbolic_ik is None:
        raise ValueError("symbolic_ik is None")

    if top:
        orientations = [[0.0, 0.0, 0.0] for _ in range(100)]
        X = center[0] + radius * np.cos(np.linspace(0, 2 * np.pi, 100))
        Y = center[1] + radius * np.sin(np.linspace(0, 2 * np.pi, 100))
        Z = center[2] * np.ones(100)
    else:
        Y = center[1] + radius * np.cos(np.linspace(0, 2 * np.pi, 100))
        Z = center[2] + radius * np.sin(np.linspace(0, 2 * np.pi, 100))
        X = center[0] * np.ones(100)
        orientations = [[0.0, -np.pi / 2, 0.0] for _ in range(100)]
        if moving_orientation:
            orientations = []
            init_rotation = R.from_euler("xyz", [0.0, -np.pi / 2, 0.0])
            for i in range(100):
                rotation = R.from_euler("xyz", [0.0, 0.0, 2 * np.pi / 100 * i])
                final_rotation = init_rotation * rotation
                orientations.append(final_rotation.as_euler("xyz"))

    previous_theta = prefered_theta
    while True:
        for i in range(100):
            goal_pose = [[X[i], Y[i], Z[i]], orientations[i]]
            print(goal_pose)
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
                        previous_theta, interval, get_joints, 0.005, goal_theta=prefered_theta
                    )
                else:
                    print("Pose not reachable________________")

            joints, elbow_position = get_joints(theta)
            previous_theta = theta
            go_to_position(placo_ik, joints, wait=0.0, arm=symbolic_ik.arm)


def make_square(
    symbolic_ik: list[SymbolicIK],
    placo_ik: IKReachyQP,
    prefered_theta: float = -5 * np.pi / 4,
) -> None:
    if symbolic_ik is None:
        raise ValueError("symbolic_ik is None")
    orientation = np.array([0.0, -np.pi / 2, 0.0])
    start_positions_r = []
    end_positions_r = []
    start_positions_l = []
    end_positions_l = []
    # if symbolic_ik.arm == "r_arm":
    print("r_arm")
    start_positions_r.append(np.array([0.3, -0.4, -0.3]))
    end_positions_r.append(np.array([0.3, -0.4, -0.0]))
    start_positions_r.append(np.array([0.3, -0.4, -0.0]))
    end_positions_r.append(np.array([0.3, -0.1, -0.0]))
    start_positions_r.append(np.array([0.3, -0.1, -0.0]))
    end_positions_r.append(np.array([0.3, -0.1, -0.3]))
    start_positions_r.append(np.array([0.3, -0.1, -0.3]))
    end_positions_r.append(np.array([0.3, -0.4, -0.3]))
    # start_positions.append(np.array([0.3, -0.1, -0.2]))
    # end_positions.append(np.array([0.3, -0.1, -0.2]))
    # else:
    print("l_arm")
    start_positions_l.append(np.array([0.3, 0.4, -0.3]))
    end_positions_l.append(np.array([0.3, 0.4, -0.0]))
    start_positions_l.append(np.array([0.3, 0.4, -0.0]))
    end_positions_l.append(np.array([0.3, 0.1, -0.0]))
    start_positions_l.append(np.array([0.3, 0.1, -0.0]))
    end_positions_l.append(np.array([0.3, 0.1, -0.3]))
    start_positions_l.append(np.array([0.3, 0.1, -0.3]))
    end_positions_l.append(np.array([0.3, 0.4, -0.3]))

    time.sleep(3)
    init = False
    previous_theta_r = prefered_theta
    previous_theta_l = np.pi - prefered_theta
    while True:
        for i in range(len(start_positions_r)):
            if i > 0:
                init = False

            previous_theta_r = make_line(
                symbolic_ik[0],
                placo_ik,
                start_positions_r[i],
                end_positions_r[i],
                orientation,
                orientation,
                prefered_theta,
                previous_theta=previous_theta_r,
                nb_points=100,
                init=init,
            )
            previous_theta_l = make_line(
                symbolic_ik[1],
                placo_ik,
                start_positions_l[i],
                end_positions_l[i],
                orientation,
                orientation,
                np.pi - prefered_theta,
                previous_theta=previous_theta_l,
                nb_points=100,
                init=init,
            )


def main_test() -> None:
    # symbolic_ik_r = SymbolicIK()

    symbolic_ik_r = SymbolicIK(
        shoulder_orientation_offset=np.array([0.0, 0.0, 15]), shoulder_position=np.array([-0.0479, -0.1913, 0.025])
    )
    symbolic_ik_l = SymbolicIK(arm="l_arm")
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

    # start_position = np.array([0.4, 0.1, -0.4])
    # end_position = np.array([0.3, -0.2, -0.1])
    # start_orientation = np.array([0.35, -1.40, 0.17])
    # end_orientation = np.array([0.0, -0.0, 0.0])
    # make_line(symbolic_ik_r, placo_ik, start_position, end_position, start_orientation, end_orientation, nb_points=300)
    prefered_theta = 5 * np.pi / 4

    make_square([symbolic_ik_r, symbolic_ik_l], placo_ik, prefered_theta=prefered_theta)
    # make_circle(symbolic_ik_r, placo_ik, prefered_theta=prefered_theta)
    # make_circle(symbolic_ik_r, placo_ik, prefered_theta=prefered_theta, center=np.array([0.2, -0.2, -0.0]), radius=0.4)
    # make_circle(
    #     symbolic_ik_r,
    #     placo_ik,
    #     prefered_theta=prefered_theta,
    #     center=np.array([0.2, -0.2, -0.0]),
    #     radius=0.4,
    #     moving_orientation=True,
    # )

    # make_circle(
    #     symbolic_ik_r, placo_ik, prefered_theta=prefered_theta, center=np.array([0.1, -0.2, -0.6]), radius=0.1, top=True
    # )
    # make_circle(
    #     symbolic_ik_r, placo_ik, prefered_theta=prefered_theta, center=np.array([0.3, -0.2, -0.4]), radius=0.1, top=True
    # )
    # make_circle(
    #     symbolic_ik_r, placo_ik, prefered_theta=prefered_theta, center=np.array([0.3, -0.4, -0.2]), radius=0.1, top=True
    # )

    # random_movement(symbolic_ik_r, placo_ik, prefered_theta=prefered_theta)

    # while True:
    #     start_position = np.array([0.4, -0.5, -0.3])
    #     end_position = np.array([0.4, -0.5, -0.0])
    #     start_orientation = np.array([0.0, -np.pi / 2, 0.0])
    #     end_orientation = np.array([0.0, -np.pi / 2, 0.0])
    #     make_line(
    #         symbolic_ik_r, placo_ik, start_position, end_position, start_orientation, end_orientation, nb_points=50, init=True
    #     )

    # current_pose = np.array([[0.4, -0.5, -0.3], [0.0, -np.pi / 2, 0.0]])
    # result = symbolic_ik_r.is_reachable(current_pose)
    # if result[0]:
    #     joints, elbow_position = result[2](result[1][0])
    #     # make_square(symbolic_ik_r, placo_ik, joints, current_pose)

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
    # else:
    #     print("Pose not reachable")


if __name__ == "__main__":
    main_test()
