import math
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt

# def get_valid_arm_joints(joints: list[float]) -> list[float]:
#     arm_joints = joints[4:7]
#     # # print(f"roll: {np.degrees(arm_joints[0])}, pitch: {np.degrees(arm_joints[1])}, yaw: {np.degrees(arm_joints[2])}")
#     rotation = R.from_euler("xyz", arm_joints, degrees=False)
#     arm_joints = rotation.as_euler("ZYZ", degrees=False)
#     # print(f"roll: {np.degrees(arm_joints[0])}, pitch: {np.degrees(arm_joints[1])}, yaw: {np.degrees(arm_joints[2])}")

#     if angle_diff(arm_joints[1], 0) > np.pi / 4:
#         arm_joints[1] = np.pi / 4
#     if angle_diff(arm_joints[1], 0) < -np.pi / 4:
#         arm_joints[1] = -np.pi / 4

#     rotation = R.from_euler("ZYZ", arm_joints, degrees=False)
#     arm_joints = rotation.as_euler("xyz", degrees=False)

#     # Quand la main est en bas, le premier angle est bien un roll naturel.
#     # Bizarrement, le second angle est bien un pitch, mais dans le sens inverse.
#     # Enfin, le dernier angle est bien un yaw naturel, lui on n'a pas besoin de le contraindre.
#     # Implémentation bête et méchante de la limitation en forme de carrée (et pas en cercle comme ça devrait être).
#     # => Suprise, ça marche pad du tout.
#     # if angle_diff(arm_joints[0], 0) > np.pi / 4:
#     #     arm_joints[0] = np.pi / 4
#     # if angle_diff(arm_joints[0], 0) < -np.pi / 4:
#     #     arm_joints[0] = -np.pi / 4
#     # if angle_diff(arm_joints[1], 0) > np.pi / 4:
#     #     arm_joints[1] = np.pi / 4
#     # if angle_diff(arm_joints[1], 0) < -np.pi / 4:
#     #     arm_joints[1] = -np.pi / 4

#     # print(f"roll: {np.degrees(arm_joints[0])}, pitch: {np.degrees(arm_joints[1])}, yaw: {np.degrees(arm_joints[2])}")
#     return [joints[0], joints[1], joints[2], joints[3]] + list(arm_joints)


def make_homogenous_matrix_from_rotation_matrix(
    position: npt.NDArray[np.float64], rotation_matrix: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Create a 4x4 homogenous matrix from a 3x3 rotation matrix and a 3x1 position vector"""
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


def limit_theta_to_interval(theta: float, previous_theta: float, interval: list[float]) -> float:
    """Limit the theta to the interval, if the theta is not in the interval, return the closest limit"""

    # Normalize the angles to be between -pi and pi
    theta = theta % (2 * np.pi)
    if theta > np.pi:
        theta -= 2 * np.pi
    previous_theta = previous_theta % (2 * np.pi)
    if previous_theta > np.pi:
        previous_theta -= 2 * np.pi

    # If the angle is in the interval, return it
    if is_valid_angle(theta, interval):
        return theta
    # If the angle is not in the interval, return the closest limit
    posDiff = angle_diff(theta, interval[1])
    negDiff = angle_diff(theta, interval[0])
    if abs(posDiff) < abs(negDiff):
        return interval[1]
    return interval[0]


def tend_to_prefered_theta(
    previous_theta: float,
    intervalle: npt.NDArray[np.float64],
    get_joints: Any,
    d_theta_max: float,
    goal_theta: float = -np.pi * 5 / 4,
) -> Tuple[bool, float]:
    """Tend to the prefered theta, if the goal_theta is not reachable, return the closest reachable theta"""
    if abs(angle_diff(goal_theta, previous_theta)) < d_theta_max:
        return True, goal_theta

    sign = angle_diff(goal_theta, previous_theta) / np.abs(angle_diff(goal_theta, previous_theta))
    return False, previous_theta + sign * d_theta_max


def get_best_continuous_theta(
    previous_theta: float,
    intervalle: npt.NDArray[np.float64],
    get_joints: Any,
    d_theta_max: float,
    prefered_theta: float,
    arm: str,
) -> Tuple[bool, float, str]:
    """Get the best continuous theta,
    if the entire circle is possible, return the prefered_theta
    theta_middle = the middle of the interval
    if theta_middle is reachable and close to the previous_theta, return theta_middle
    if theta_middle is reachable but far from the previous_theta, return the closest theta to the previous_theta
    if theta_middle is not reachable and previous_theta is okay return previous_theta
    if theta_middle is not reachable and previous_theta is not okay, return the closest theta to the prefered_theta"""
    side = 1
    if arm == "l_arm":
        side = -1

    state = f"{arm}"
    state += "\n" + f"intervalle: {intervalle}"
    epsilon = 0.00001
    if (abs(abs(intervalle[0]) + abs(intervalle[1]) - 2 * np.pi)) < epsilon:
        # The entire circle is possible, we'll aim for prefered_theta
        state += "\n" + "All the circle is possible."
        theta_middle = prefered_theta
    else:
        if intervalle[0] > intervalle[1]:
            theta_middle = (intervalle[0] + intervalle[1]) / 2 - np.pi
        else:
            theta_middle = (intervalle[0] + intervalle[1]) / 2

    state += "\n" + f"theta milieu {theta_middle}"
    state += "\n" + f"angle diff {angle_diff(theta_middle, previous_theta)}"

    joints, elbow_position = get_joints(theta_middle)

    if is_elbow_ok(elbow_position, side):
        if abs(angle_diff(theta_middle, previous_theta)) < d_theta_max:
            # middle theta is reachable and close to previous theta
            state += "\n" + "theta milieu ok et proche"
            return True, theta_middle, state
        else:
            # middle theta is reachable but far from previous theta
            sign = angle_diff(theta_middle, previous_theta) / np.abs(angle_diff(theta_middle, previous_theta))
            state += "\n" + f"sign = {sign}"

            # if perf needed delete this and return False, (previous_theta + sign * d_theta_max)
            theta_side = previous_theta + sign * d_theta_max
            joints, elbow_position = get_joints(theta_side)
            is_reachable = is_elbow_ok(elbow_position, side)
            state += "\n" + f"theta milieu ok mais loin - et moi je suis {is_reachable}"
            return is_reachable, theta_side, state

    else:
        joints, elbow_position = get_joints(previous_theta)
        is_reachable = is_elbow_ok(elbow_position, side)
        if is_reachable:
            # middle theta is not reachable but previous theta is okay
            state += "\n" + "theta milieu pas ok mais moi ok - bouge pas "
            return True, previous_theta, state
        else:
            # middle theta is not reachable and previous theta is not okay
            if abs(angle_diff(prefered_theta, previous_theta)) < d_theta_max:
                # prefered theta is close to previous theta
                state += "\n" + "theta milieu pas ok et moi pas ok - proche de theta pref"
                return False, prefered_theta, state
            # prefered theta is far from previous theta
            sign = angle_diff(prefered_theta, previous_theta) / np.abs(angle_diff(prefered_theta, previous_theta))
            state += "\n" + "theta milieu pas ok et moi pas ok - bouge vers theta pref"
            return False, previous_theta + sign * d_theta_max, state


def is_elbow_ok(elbow_position: npt.NDArray[np.float64], side: int) -> bool:
    """Check if the elbow is in a valid position
    Prevent the elbow to touch the robot body"""
    return bool(elbow_position[1] * side < -0.2)


def is_valid_angle(angle: float, intervalle: list[float]) -> bool:
    """Check if an angle is in the intervalle"""
    if intervalle[0] % (2 * np.pi) == intervalle[1] % (2 * np.pi):
        return True
    if intervalle[0] < intervalle[1]:
        return (intervalle[0] <= angle) and (angle <= intervalle[1])
    return (intervalle[0] <= angle) or (angle <= intervalle[1])


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
    """Show a circle in the 3D space"""
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
    """Show a sphere in the 3D space"""
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]  # type: ignore
    x = np.cos(u) * np.sin(v) * radius + center[0]
    y = np.sin(u) * np.sin(v) * radius + center[1]
    z = np.cos(v) * radius + center[2]
    ax.plot_wireframe(x, y, z, color=color, alpha=0.2)


def show_point(ax: Any, point: npt.NDArray[np.float64], color: str) -> None:
    """Show a point in the 3D space"""
    ax.plot(point[0], point[1], point[2], "o", color=color)
