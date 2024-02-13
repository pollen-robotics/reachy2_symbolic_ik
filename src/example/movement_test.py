from pathlib import Path

import numpy as np
import numpy.typing as npt
from reachy_placo.ik_reachy_placo import IKReachyQP

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils_placo import go_to_position


def make_movement_test(symbolic_ik: SymbolicIK, placo_ik: IKReachyQP, goal_pose: npt.NDArray[np.float64]) -> None:
    result = symbolic_ik.is_reachable(goal_pose)
    if result[0]:
        print(int((result[1][1] - result[1][0]) * 50))
        angles = np.linspace(result[1][0], result[1][1], int((result[1][1] - result[1][0]) * 50))
        while True:
            for angle in angles:
                joints = result[2](angle)
                go_to_position(placo_ik, joints, wait=0.0)
    else:
        print("Pose not reachable")


def make_line(
    symbolic_ik: SymbolicIK,
    placo_ik: IKReachyQP,
    start_position: npt.NDArray[np.float64],
    end_position: npt.NDArray[np.float64],
    start_orientation: npt.NDArray[np.float64],
    end_orientation: npt.NDArray[np.float64],
    nb_points: int = 100,
) -> None:
    x = np.linspace(start_position[0], end_position[0], nb_points)
    y = np.linspace(start_position[1], end_position[1], nb_points)
    z = np.linspace(start_position[2], end_position[2], nb_points)
    roll = np.linspace(start_orientation[0], end_orientation[0], nb_points)
    pitch = np.linspace(start_orientation[1], end_orientation[1], nb_points)
    yaw = np.linspace(start_orientation[2], end_orientation[2], nb_points)
    for i in range(nb_points):
        goal_pose = [[x[i], y[i], z[i]], [roll[i], pitch[i], yaw[i]]]
        result = symbolic_ik.is_reachable(goal_pose)
        if result[0]:
            angle = np.linspace(result[1][0], result[1][1], 3)[1]
            joints = result[2](angle)
            go_to_position(placo_ik, joints, wait=0.0)
        else:
            print("Pose not reachable")


def main_test() -> None:
    symbolic_ik = SymbolicIK()
    urdf_path = Path("src/config_files")
    for file in urdf_path.glob("**/*.urdf"):
        if file.stem == "reachy2":
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

    goal_position = [0.20, -0.2, -0.0]
    goal_orientation = np.array([-20, -60, 10])
    goal_orientation = np.deg2rad(goal_orientation)
    goal_pose = np.array([goal_position, goal_orientation])
    make_movement_test(symbolic_ik, placo_ik, goal_pose)

    # start_position = np.array([0.4, 0.1, -0.4])
    # end_position = np.array([0.3, -0.2, -0.1])
    # start_orientation = np.array([0.35, -1.40, 0.17])
    # end_orientation = np.array([0.0, -0.0, 0.0])
    # make_line(symbolic_ik, placo_ik, start_position, end_position, start_orientation, end_orientation, nb_points=300)


if __name__ == "__main__":
    main_test()
