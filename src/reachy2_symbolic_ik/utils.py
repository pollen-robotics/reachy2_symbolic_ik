from typing import Any, Tuple

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


def rotation_matrix_from_vectors(vect1: npt.NDArray[np.float64], vect2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Find the rotation matrix that aligns vect1 to vect2
    :param vect1: A 3d "source" vector
    :param vect2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vect1, aligns it with vect2.
    """

    a, b = (vect1 / np.linalg.norm(vect1)).reshape(3), (vect2 / np.linalg.norm(vect2)).reshape(3)
    v = np.cross(a, b)
    if np.all(np.isclose(a, b)):
        return np.eye(3)
    if np.all(a == -b):
        matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        return matrix

    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.array(np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2)))
    return rotation_matrix


def shoulder_limits(intervalle: npt.NDArray[np.float64], get_joints: Any, arm: str = "r_arm") -> Tuple[bool, float]:
    theta = float(np.linspace(intervalle[0], intervalle[1], 3)[1])
    joints, elbow_position = get_joints(theta)
    # print(elbow_position)
    # print(joints)
    # print(joints[1])
    side = 1
    if arm == "l_arm":
        side = -1
    # print(elbow_position[1] * side)
    if elbow_position[1] * side > -0.2:
        joints0, elbow_position0 = get_joints(intervalle[0])
        joints1, elbow_position1 = get_joints(intervalle[1])
        if (elbow_position0[1] > -0.2) and (elbow_position1[1] > -0.2):
            return False, 0.0
        elif elbow_position0[1] * side > -0.2:
            thetas = np.linspace(theta, intervalle[1], 100)
            return find_theta(thetas, get_joints, side)
        elif elbow_position1[1] * side > -0.2:
            thetas = np.linspace(intervalle[0], theta, 100)
            return find_theta(thetas, get_joints, side)
        else:
            if elbow_position0[2] * side < elbow_position1[2]:
                thetas = np.linspace(intervalle[0], theta, 100)
            else:
                thetas = np.linspace(theta, intervalle[1], 100)
            return find_theta(thetas, get_joints, side)
    return True, theta


def find_theta(thetas: npt.NDArray[np.float64], get_joints: Any, side: int) -> Tuple[bool, float]:
    for theta in thetas:
        joints, elbow_position = get_joints(theta)
        if elbow_position[1] * side <= -0.2:
            return True, theta
    return False, 0.0


def show_point(ax: Any, point: npt.NDArray[np.float64], color: str) -> None:
    ax.plot(point[0], point[1], point[2], "o", color=color)


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
    Rmat = rotation_matrix_from_vectors(np.array([1.0, 0.0, 0.0]), np.array(normal_vector))
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
