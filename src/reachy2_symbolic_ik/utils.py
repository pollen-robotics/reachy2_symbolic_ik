import math
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt


def make_homogenous_matrix_from_rotation_matrix(
    position: npt.NDArray[np.float64], rotation_matrix: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.array(
        [
            [rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], position[0]],
            [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], position[1]],
            [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rotation_matrix_from_vector(vect: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Find the rotation matrix that aligns vect1 to vect
    :param vect1: A 3d "source" vector
    :param vect: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vect1, aligns it with vect.
    """
    vect1 = np.array([1, 0, 0])
    vect2 = (vect / np.linalg.norm(vect)).reshape(3)

    # handling cross product colinear
    if np.all(np.isclose(vect1, vect2)):
        return np.eye(3)

    # handling cross product colinear
    if np.all(np.isclose(vect1, -vect2)):
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    v = np.cross(vect1, vect2)
    c = np.dot(vect1, vect2)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.array(np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2)))
    return rotation_matrix


def get_theta_from_current_pose(
    get_joints: Any,
    intervalle: npt.NDArray[np.float64],
    current_joints: List[float],
    joints_tolerance: List[float],
    nb_points: int,
) -> Tuple[bool, float]:
    thetas = np.linspace(intervalle[0], intervalle[1], 360)
    # side = 1
    # if arm == "l_arm":
    #     side = -1
    d_min = 1000.0
    theta_min = 0.0
    joints_min = []
    for theta in thetas:
        joints, elbow_position = get_joints(theta)
        d = float(np.linalg.norm(np.array(joints) - np.array(current_joints)))
        if d < d_min:
            print(d)
            d_min = d
            theta_min = theta
            joints_min = joints
    for joint in joints_min:
        if abs(joint - current_joints[0]) > joints_tolerance[0]:
            return False, theta_min
    return True, theta_min


def tend_to_prefered_theta(
    previous_theta: float,
    intervalle: npt.NDArray[np.float64],
    get_joints: Any,
    d_theta_max: float,
    arm: str,
    goal_theta: float = -np.pi * 5 / 4,
) -> Tuple[bool, float]:
    side = 1
    if arm == "l_arm":
        side = -1

    if abs(angle_diff(goal_theta, previous_theta)) < d_theta_max:
        print(f"diff ok {angle_diff(previous_theta, goal_theta * side)}")
        return True, goal_theta

    if angle_diff(goal_theta, previous_theta) == 0:
        print("---------------------------tend to -------------------------")

    sign = angle_diff(goal_theta, previous_theta) / np.abs(angle_diff(goal_theta, previous_theta))

    print(f"tend to go to goal theta {side} {previous_theta} -- {previous_theta + sign * d_theta_max}")
    return False, previous_theta + sign * d_theta_max


def get_best_continuous_theta(
    previous_theta: float,
    intervalle: npt.NDArray[np.float64],
    get_joints: Any,
    d_theta_max: float,
    prefered_theta: float,
    arm: str,
) -> Tuple[bool, float]:
    side = 1
    if arm == "l_arm":
        side = -1

    if angle_diff(intervalle[0], intervalle[1]) > 0:
        print("OMG ANGLE DIFF > 0 ")
        theta_middle = angle_diff(intervalle[0], intervalle[1]) / 2 + intervalle[1] + np.pi
    else:
        theta_middle = angle_diff(intervalle[0], intervalle[1]) / 2 + intervalle[1]
    print(f"theta milieu {theta_middle}")

    print(f"angle diff {angle_diff(theta_middle, previous_theta)}")

    joints, elbow_position = get_joints(theta_middle)

    if is_elbow_ok(elbow_position, side):
        if abs(angle_diff(theta_middle, previous_theta)) < d_theta_max:
            print("theta milieu ok et proche")
            return True, theta_middle
        else:
            if angle_diff(theta_middle, previous_theta) == 0:
                print("--------------------------- get best theta -------------------------")
            sign = angle_diff(theta_middle, previous_theta) / np.abs(angle_diff(theta_middle, previous_theta))

            # if perf needed delete this and return False, (previous_theta + sign * d_theta_max)
            theta_side = previous_theta + sign * d_theta_max
            joints, elbow_position = get_joints(theta_side)
            is_reachable = is_elbow_ok(elbow_position, side)
            print(f"theta milieu ok mais loin - et moi je suis {is_reachable}")
            return is_reachable, theta_side

    else:
        joints, elbow_position = get_joints(previous_theta)
        is_reachable = is_elbow_ok(elbow_position, side)
        if is_reachable:
            print("theta milieu pas ok mais moi ok - bouge pas ")
            return True, previous_theta
        else:
            if angle_diff(prefered_theta, previous_theta) == 0:
                print("--------------------------- get best theta 1 -------------------------")
            sign = angle_diff(prefered_theta, previous_theta) / np.abs(angle_diff(prefered_theta, previous_theta))
            print("theta milieu pas ok et moi pas ok - bouge vers theta pref")
            return False, previous_theta + sign * d_theta_max

    # in d_theta interval - close to theta middle
    # if abs(angle_diff(theta_middle, previous_theta)) < d_theta_max:
    #     theta = theta_middle

    #     joints, elbow_position = get_joints(theta)

    # #

    # print("theta hors limites")
    # return False, previous_theta + sign * d_theta_max


def is_elbow_ok(elbow_position: npt.NDArray[np.float64], side: int) -> bool:
    return bool(elbow_position[1] * side < -0.2)


def find_theta(thetas: npt.NDArray[np.float64], get_joints: Any, side: int) -> Tuple[bool, float]:
    for theta in thetas:
        joints, elbow_position = get_joints(theta)
        if elbow_position[1] * side <= -0.2:
            return True, theta
    return False, 0.0


def show_point(ax: Any, point: npt.NDArray[np.float64], color: str) -> None:
    ax.plot(point[0], point[1], point[2], "o", color=color)


def angle_diff(a: float, b: float) -> float:
    """Returns the smallest distance between 2 angles"""
    d = a - b
    d = ((d + math.pi) % (2 * math.pi)) - math.pi
    return d


def show_circle(
    ax: Any,
    center: npt.NDArray[np.float64],
    radius: float,
    normal_vector: npt.NDArray[np.float64],
    intervalles: npt.NDArray[np.float64],
    color: str,
) -> None:
    theta = []
    for intervalle in intervalles:
        angle = np.linspace(intervalle[0], intervalle[1], 100)
        for a in angle:
            theta.append(a)

    y = radius * np.cos(theta)
    z = radius * np.sin(theta)
    x = np.zeros(len(theta))
    Rmat = rotation_matrix_from_vector(np.array(normal_vector))
    Tmat = make_homogenous_matrix_from_rotation_matrix(center, Rmat)

    x2 = np.zeros(len(theta))
    y2 = np.zeros(len(theta))
    z2 = np.zeros(len(theta))
    for k in range(len(theta)):
        p = [x[k], y[k], z[k], 1]
        p2 = np.dot(Tmat, p)
        x2[k] = p2[0]
        y2[k] = p2[1]
        z2[k] = p2[2]
    ax.plot(center[0], center[1], center[2], "o", color=color)
    ax.plot(x2, y2, z2, color)


def show_sphere(ax: Any, center: npt.NDArray[np.float64], radius: np.float64, color: str) -> None:
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]  # type: ignore
    x = np.cos(u) * np.sin(v) * radius + center[0]
    y = np.sin(u) * np.sin(v) * radius + center[1]
    z = np.cos(v) * radius + center[2]
    ax.plot_wireframe(x, y, z, color=color, alpha=0.2)
