import time

import numpy as np

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import (
    make_homogenous_matrix_from_rotation_matrix,
    rotation_matrix_from_vector,
)


def test_is_reachable(ik: SymbolicIK) -> None:
    goal_position = [0.3, -0.1, 0.1]
    goal_orientation = [np.radians(20), np.radians(-50), np.radians(20)]
    goal_pose = np.array([goal_position, goal_orientation])
    start_time = time.time()
    for i in range(10000):
        ik.is_reachable(goal_pose)
    end_time = time.time()
    print("is_reachable : ", end_time - start_time)


def test_get_joints(ik: SymbolicIK) -> None:
    goal_position = [0.3, -0.1, 0.1]
    goal_orientation = [np.radians(20), np.radians(-50), np.radians(20)]
    goal_pose = np.array([goal_position, goal_orientation])
    result = ik.is_reachable(goal_pose)
    alpha = result[1][0]
    start_time = time.time()
    for i in range(10000):
        ik.get_joints(alpha)
    end_time = time.time()
    print("get_joints : ", end_time - start_time)


def test_get_wrist_position(ik: SymbolicIK) -> None:
    goal_position = [0.3, -0.1, 0.1]
    goal_orientation = [np.radians(20), np.radians(-50), np.radians(20)]
    goal_pose = np.array([goal_position, goal_orientation])
    start_time = time.time()
    for i in range(10000):
        ik.get_wrist_position(goal_pose)
    end_time = time.time()
    print("get_wrist_position : ", end_time - start_time)


def test_get_limitation_wrist_circle(ik: SymbolicIK) -> None:
    goal_position = [0.3, -0.1, 0.1]
    goal_orientation = [np.radians(20), np.radians(-50), np.radians(20)]
    goal_pose = np.array([goal_position, goal_orientation])
    ik.wrist_position = ik.get_wrist_position(goal_pose)
    start_time = time.time()
    for i in range(10000):
        ik.get_limitation_wrist_circle(goal_pose)
    end_time = time.time()
    print("get_limitation_wrist_circle : ", end_time - start_time)
    pass


def test_get_intersection_circle(ik: SymbolicIK) -> None:
    goal_position = [0.3, -0.1, 0.1]
    goal_orientation = [np.radians(20), np.radians(-50), np.radians(20)]
    goal_pose = np.array([goal_position, goal_orientation])
    ik.wrist_position = ik.get_wrist_position(goal_pose)
    start_time = time.time()
    for i in range(10000):
        ik.get_intersection_circle(goal_pose)
    end_time = time.time()
    print("get_intersection_circle : ", end_time - start_time)
    pass


def test_are_cricles_linked(ik: SymbolicIK) -> None:
    goal_position = [0.3, -0.1, 0.1]
    goal_orientation = [np.radians(20), np.radians(-50), np.radians(20)]
    goal_pose = np.array([goal_position, goal_orientation])
    ik.wrist_position = ik.get_wrist_position(goal_pose)
    intersection_circle = ik.get_intersection_circle(goal_pose)
    limitation_wrist_circle = ik.get_limitation_wrist_circle(goal_pose)
    start_time = time.time()
    for i in range(10000):
        ik.are_circles_linked(intersection_circle, limitation_wrist_circle)
    end_time = time.time()
    print("are_circles_linked : ", end_time - start_time)


def test_rotation_matrix_from_vector() -> None:
    goal_position = [0.3, -0.1, 0.1]
    goal_orientation = [np.radians(20), np.radians(-50), np.radians(20)]
    goal_pose = np.array([goal_position, goal_orientation])
    ik.wrist_position = ik.get_wrist_position(goal_pose)
    limitation_wrist_circle = ik.get_limitation_wrist_circle(goal_pose)
    start_time = time.time()
    for i in range(10000):
        rotation_matrix_from_vector(limitation_wrist_circle[2])
    end_time = time.time()
    print("rotation_matrix_from_vector : ", end_time - start_time)


def test_points_of_nearest_point(ik: SymbolicIK) -> None:
    goal_position = [0.3, -0.1, 0.1]
    goal_orientation = [np.radians(20), np.radians(-50), np.radians(20)]
    goal_pose = np.array([goal_position, goal_orientation])
    ik.wrist_position = ik.get_wrist_position(goal_pose)
    intersection_circle = ik.get_intersection_circle(goal_pose)
    limitation_wrist_circle = ik.get_limitation_wrist_circle(goal_pose)
    start_time = time.time()
    for i in range(10000):
        ik.points_of_nearest_approach(
            limitation_wrist_circle[0], limitation_wrist_circle[2], intersection_circle[0], intersection_circle[2]
        )
    end_time = time.time()
    print("points_of_nearest_approach", end_time - start_time)


def test_intersection_circle_line_3d(ik: SymbolicIK) -> None:
    goal_position = [0.3, -0.1, 0.1]
    goal_orientation = [np.radians(20), np.radians(-50), np.radians(20)]
    goal_pose = np.array([goal_position, goal_orientation])
    ik.wrist_position = ik.get_wrist_position(goal_pose)
    intersection_circle = ik.get_intersection_circle(goal_pose)
    limitation_wrist_circle = ik.get_limitation_wrist_circle(goal_pose)
    q, v = ik.points_of_nearest_approach(
        limitation_wrist_circle[0], limitation_wrist_circle[2], intersection_circle[0], intersection_circle[2]
    )

    start_time = time.time()
    for i in range(10000):
        ik.intersection_circle_line_3d_vd(limitation_wrist_circle[0], limitation_wrist_circle[1], v, q)
    end_time = time.time()
    print("intersection_circle_line_3d_vd : ", end_time - start_time)


def test_make_homogenous_matrix_from_rotation_matrix() -> None:
    position = np.array([0.3, -0.1, 0.1])
    rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    start_time = time.time()
    for i in range(10000):
        make_homogenous_matrix_from_rotation_matrix(position, rotation_matrix)
    end_time = time.time()
    print("make_homogenous_matrix_from_rotation_matrix : ", end_time - start_time)


if __name__ == "__main__":
    ik = SymbolicIK()
    test_is_reachable(ik)
    test_get_joints(ik)
    test_get_wrist_position(ik)
    test_get_limitation_wrist_circle(ik)
    test_get_intersection_circle(ik)
    test_are_cricles_linked(ik)
    test_rotation_matrix_from_vector()
    test_points_of_nearest_point(ik)
    test_intersection_circle_line_3d(ik)
    test_make_homogenous_matrix_from_rotation_matrix()
    print("All tests passed!")
