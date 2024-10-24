import copy
import math
import xml.etree.ElementTree as ET
from io import StringIO
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R


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


def get_singularity_position(
    arm: str,
    shoulder_position: npt.NDArray[np.float64],
    shoulder_offset: npt.NDArray[np.float64],
    upper_arm_size: float,
    forearm_size: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if arm == "r_arm":
        side = 1
    else:
        side = -1
    rotation_matrix = R.from_euler("xyz", shoulder_offset, degrees=True).as_matrix()
    T_torso_shoulder = make_homogenous_matrix_from_rotation_matrix(shoulder_position, rotation_matrix)
    elbow_position = np.array([0.0, -upper_arm_size * side, 0.0, 1.0])
    elbow_singularity_position = np.dot(T_torso_shoulder, elbow_position)[:3]
    wrist_position = np.array([0.0, -(upper_arm_size + forearm_size) * side, 0.0, 1.0])
    wrist_singularity_position = np.dot(T_torso_shoulder, wrist_position)[:3]
    return elbow_singularity_position, wrist_singularity_position


def distance_from_singularity(
    elbow_position: npt.NDArray[np.float64],
    arm: str,
    shoulder_position: npt.NDArray[np.float64],
    shoulder_offset: npt.NDArray[np.float64],
    upper_arm_size: float,
    forearm_size: float,
) -> float:
    """Compute the distance from the singularity"""
    singularity_position, _ = get_singularity_position(arm, shoulder_position, shoulder_offset, upper_arm_size, forearm_size)
    return float(np.linalg.norm(elbow_position - singularity_position))


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


def get_euler_from_homogeneous_matrix(
    homogeneous_matrix: npt.NDArray[np.float64], degrees: bool = False
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    position = homogeneous_matrix[:3, 3]
    rotation_matrix = homogeneous_matrix[:3, :3]
    euler_angles = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=degrees)
    return position, euler_angles


def limit_theta_to_interval(theta: float, previous_theta: float, interval: npt.NDArray[np.float64]) -> Tuple[float, str]:
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
        return theta, "theta in interval"
    # If the angle is not in the interval, return the closest limit
    posDiff = angle_diff(theta, interval[1])
    negDiff = angle_diff(theta, interval[0])
    if abs(posDiff) < abs(negDiff):
        return interval[1], "theta not in interval"
    return interval[0], "theta not in interval"


def tend_to_preferred_theta(
    previous_theta: float,
    interval: npt.NDArray[np.float64],
    get_joints: Any,
    d_theta_max: float,
    goal_theta: float = -np.pi * 5 / 4,
) -> Tuple[bool, float]:
    """Tend to the preferred theta, if the goal_theta is not reachable, return the closest reachable theta"""
    if abs(angle_diff(goal_theta, previous_theta)) < d_theta_max:
        return True, goal_theta

    sign = angle_diff(goal_theta, previous_theta) / np.abs(angle_diff(goal_theta, previous_theta))
    return False, previous_theta + sign * d_theta_max


def get_best_continuous_theta(
    previous_theta: float,
    interval: npt.NDArray[np.float64],
    # get_joints: Any,
    get_elbow_position: Any,
    d_theta_max: float,
    preferred_theta: float,
    arm: str,
    singularity_offset: float,
    singularity_limit_coeff: float,
    elbow_singularity_position: npt.NDArray[np.float64],
) -> Tuple[bool, float, str]:
    """Get the best continuous theta,
    if the entire circle is possible, return the preferred_theta
    theta_middle = the middle of the interval
    if theta_middle is reachable and close to the previous_theta, return theta_middle
    if theta_middle is reachable but far from the previous_theta, return the closest theta to the previous_theta
    if theta_middle is not reachable and previous_theta is okay return previous_theta
    if theta_middle is not reachable and previous_theta is not okay, return the closest theta to the preferred_theta"""
    side = 1
    if arm == "l_arm":
        side = -1

    state = f"{arm}"
    state += "\n" + f"interval: {interval}"
    epsilon = 0.00001
    if (abs(abs(interval[0]) + abs(interval[1]) - 2 * np.pi)) < epsilon:
        # The entire circle is possible, we'll aim for preferred_theta
        state += "\n" + "All the circle is possible."
        theta_middle = preferred_theta
    else:
        if interval[0] > interval[1]:
            theta_middle = (interval[0] + interval[1]) / 2 - np.pi
        else:
            theta_middle = (interval[0] + interval[1]) / 2

    state += "\n" + f"theta milieu {theta_middle}"
    state += "\n" + f"angle diff {angle_diff(theta_middle, previous_theta)}"

    elbow_position = get_elbow_position(theta_middle)
    # states = f"elbow_position: {elbow_position} : {is_elbow_ok(elbow_position, side)}"

    if is_elbow_ok(elbow_position, side, singularity_offset, singularity_limit_coeff, elbow_singularity_position):
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
            # joints, elbow_position = get_joints(theta_side)

            elbow_position = get_elbow_position(theta_side)
            is_reachable = is_elbow_ok(
                elbow_position, side, singularity_offset, singularity_limit_coeff, elbow_singularity_position
            )
            is_reachable = is_reachable and is_valid_angle(theta_side, interval)
            state += "\n" + f"previous_theta: {previous_theta}"
            state += "\n" + f"theta milieu ok mais loin - et moi je suis {is_reachable}"
            # if not is_reachable:
            #     state += "\n tend to preferred theta __________"
            #     _, theta = tend_to_preferred_theta(previous_theta, interval, get_joints, d_theta_max, preferred_theta)
            #     return False, theta, state
            return is_reachable, theta_side, state

    else:
        elbow_position = get_elbow_position(previous_theta)
        is_reachable = is_elbow_ok(
            elbow_position, side, singularity_offset, singularity_limit_coeff, elbow_singularity_position
        )
        if is_reachable:
            # middle theta is not reachable but previous theta is okay
            state += "\n" + "theta milieu pas ok mais moi ok - bouge pas "
            return True, previous_theta, state
        else:
            # middle theta is not reachable and previous theta is not okay
            if abs(angle_diff(preferred_theta, previous_theta)) < d_theta_max:
                # preferred theta is close to previous theta
                state += "\n" + "theta milieu pas ok et moi pas ok - proche de theta pref"
                return False, preferred_theta, state
            # preferred theta is far from previous theta
            sign = angle_diff(preferred_theta, previous_theta) / np.abs(angle_diff(preferred_theta, previous_theta))
            state += "\n" + "theta milieu pas ok et moi pas ok - bouge vers theta pref"
            return False, previous_theta + sign * d_theta_max, state


def get_best_continuous_theta2(
    previous_theta: float,
    interval: npt.NDArray[np.float64],
    get_elbow_position: Any,
    nb_search_points: int,
    d_theta_max: float,
    preferred_theta: float,
    arm: str,
    singularity_offset: float,
    singularity_limit_coeff: float,
    elbow_singularity_position: npt.NDArray[np.float64],
) -> Tuple[bool, float, str]:
    """Get the best theta to aim for,
    tend to the closest reachable theta (sampled with nb_search_points) to preferred_theta"""
    state = f"{arm}"
    state += "\n" + f"interval: {interval}"
    is_reachable, theta_goal, state = get_best_discrete_theta(
        previous_theta,
        interval,
        get_elbow_position,
        nb_search_points,
        preferred_theta,
        arm,
        singularity_offset,
        singularity_limit_coeff,
        elbow_singularity_position,
    )
    if not is_reachable:
        # No solution was found
        return False, previous_theta, state

    # A solution was found
    if abs(angle_diff(theta_goal, previous_theta)) < d_theta_max:
        # theta_goal is reachable and close to previous theta
        state += "\n" + "theta theta_goal ok et proche"
        return True, theta_goal, state
    else:
        # middle theta is reachable but far from previous theta
        sign = angle_diff(theta_goal, previous_theta) / np.abs(angle_diff(theta_goal, previous_theta))
        # state += "\n" + "theta theta_goal ok mais loin"
        state = "\n" + "theta theta_goal ok mais loin"
        theta_tends = previous_theta + sign * d_theta_max
        # Saying True here is not always true. It could be that the intermediate theta_tends is not reachable,
        # but eventually it will reach a reachable theta
        return True, theta_tends, state


def get_best_theta_to_current_joints(
    get_joints: Any, nb_search_points: int, current_joints: list[float], arm: str, preferred_theta: float
) -> Tuple[float, str]:
    """Searches all theta in the entire circle that minimises the distance to the current joints."""
    best_theta = None
    best_distance = np.inf
    state = ""
    current_joints = copy.deepcopy(current_joints)
    # # modulo 2pi on all joints
    # for i in range(len(current_joints)):
    #     current_joints[i] = ((current_joints[i] + np.pi) % (2 * np.pi)) - np.pi

    # Simple linear search
    # for theta in np.linspace(-np.pi, np.pi, nb_search_points):
    #     joints, elbow_position = get_joints(theta)
    #     distance = np.linalg.norm(joints - current_joints)
    #     if distance < best_distance:
    #         best_theta = theta
    #         best_distance = distance

    # Dichotomic search to find the best theta instead
    low = -np.pi
    high = np.pi
    if arm == "l_arm":
        low = 0
        high = 2 * np.pi

    tolerance = 0.01

    joints, elbow_position = get_joints(preferred_theta)
    diff = np.linalg.norm([angle_diff(joints[i], current_joints[i]) for i in range(len(current_joints))])
    # state += f" \n diff = {diff}"
    if diff < tolerance:
        return preferred_theta, f"preferred_theta worked! \n joints = {joints} \n current_joints = {current_joints}"

    while (high - low) > tolerance:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3
        joints1, elbow_position1 = get_joints(mid1)
        joints2, elbow_position2 = get_joints(mid2)
        diff1 = [angle_diff(joints1[i], current_joints[i]) for i in range(len(current_joints))]
        diff2 = [angle_diff(joints2[i], current_joints[i]) for i in range(len(current_joints))]
        f_mid1 = np.linalg.norm(diff1)
        f_mid2 = np.linalg.norm(diff2)

        # mid = (low + high) / 2
        # state += f" \n mid1 = {mid1}, mid2 = {mid2}"
        # state += f" \n f_mid1 = {f_mid1}, f_mid2 = {f_mid2}"

        if f_mid1 < f_mid2:
            high = mid2
        else:
            low = mid1

    state += f" \n low = {low}, high = {high}"

    best_theta = (low + high) / 2
    joints, _ = get_joints(best_theta)
    best_distance = np.linalg.norm(joints - current_joints)  # type: ignore
    state += f" \n best_distance = {best_distance}"
    state += f" \n joints = {joints}"
    state += f" \n current_joints = {current_joints}"

    # print(f"best_theta = {best_theta}")
    return best_theta, state


def get_best_discrete_theta(
    previous_theta: float,
    interval: npt.NDArray[np.float64],
    get_elbow_position: Any,
    nb_search_points: int,
    preferred_theta: float,
    arm: str,
    singularity_offset: float,
    singularity_limit_coeff: float,
    elbow_singularity_position: npt.NDArray[np.float64],
) -> Tuple[bool, float, str]:
    """Searches a valid theta in the interval that is the closest to preferred_theta.
    A valid theta is a theta that is reachable and does not make the elbow touch the robot body."""
    side = 1
    if arm == "l_arm":
        side = -1

    state = f"{arm}"
    state += "\n" + f"interval: {interval}, preferred_theta: {preferred_theta}"
    epsilon = 0.00001
    best_theta = None
    best_distance = np.inf

    if is_valid_angle(preferred_theta, interval):
        # if preferred_theta is in the interval, test it first
        elbow_position = get_elbow_position(preferred_theta)
        if is_elbow_ok(elbow_position, side, singularity_offset, singularity_limit_coeff, elbow_singularity_position):
            best_theta = preferred_theta
            best_distance = 0
            state += "\n" + "preferred_theta worked!"
            return True, best_theta, state

    if (abs(abs(interval[0]) + abs(interval[1]) - 2 * np.pi)) < epsilon:
        # The entire circle is possible, sampling with a vertical symmetry (instead of horizontal)
        # so that the results are symetric for both arms
        theta_points = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, nb_search_points)
    else:
        # Sampling the interval
        if interval[0] < interval[1]:
            theta_points = np.linspace(interval[0], interval[1], nb_search_points)
        else:
            theta_points = np.linspace(interval[0], interval[1] + 2 * np.pi, nb_search_points)

    state += "\n" + f"theta_points: {theta_points}"
    debug_dict = {}

    # test all theta points and choose the closest to preferred_theta
    for theta in theta_points:
        elbow_position = get_elbow_position(theta)
        if is_elbow_ok(elbow_position, side, singularity_offset, singularity_limit_coeff, elbow_singularity_position):
            distance = abs(angle_diff(theta, preferred_theta))
            debug_dict[theta] = distance
            if distance < best_distance:
                best_theta = theta
                best_distance = distance
        else:
            debug_dict[theta] = np.inf
    state += "\n" + f"debug_dict: {debug_dict}"

    if best_theta is not None:
        return True, best_theta, state
    else:
        return False, previous_theta, state


# To be recoded base on get_best_discrete_theta
# def get_best_discrete_theta_min_mouvement(
#     previous_theta: float,
#     interval: npt.NDArray[np.float64],
#     get_joints: Any,
#     nb_search_points: int,
#     preferred_theta: float,
#     arm: str,
#     current_joints: npt.NDArray[np.float64],
# ) -> Tuple[bool, float, str]:
#     """Searches a valid theta in the interval that minimises the mouvement in joint space.
#     A valid theta is a theta that is reachable and does not make the elbow touch the robot body."""
#     side = 1
#     if arm == "l_arm":
#         side = -1

#     state = f"{arm}"
#     state += "\n" + f"interval: {interval}"
#     epsilon = 0.00001
#     # get nb_search_points points in the interval
#     if interval[0] < interval[1]:
#         theta_points = np.linspace(interval[0], interval[1], nb_search_points)
#     else:
#         theta_points = np.linspace(interval[1], interval[0] + 2 * np.pi, nb_search_points)

#     best_theta = None
#     best_distance = np.inf
#     for theta in theta_points:
#         joints, elbow_position = get_joints(theta)
#         if is_elbow_ok(elbow_position, side):
#             # test all theta points and rank them by the l2 distance to the current joints
#             # distance = np.linalg.norm(joints - current_joints)
#             # # test all theta points and rank them by the max distance to the current joints
#             distance = np.max(np.abs(joints - current_joints))
#             if distance < best_distance:
#                 best_theta = theta
#                 best_distance = distance

#     if best_theta is not None:
#         return True, best_theta, state
#     else:
#         return False, previous_theta, state


def is_elbow_ok(
    elbow_position: npt.NDArray[np.float64],
    side: int,
    singularity_offset: float,
    singularity_limit_coeff: float,
    elbow_singularity_position: npt.NDArray[np.float64],
) -> bool:
    """Check if the elbow is in a valid position
    Prevent the elbow to touch the robot body"""
    is_ok = True
    if elbow_position[1] * side > -0.15:
        if elbow_position[0] < 0.15:
            is_ok = False
    # ultra safe config
    is_ok = elbow_position[1] * side < -0.2
    # if elbow_position[0] > elbow_singularity_position[0]:
    is_ok = is_ok and (
        elbow_position[2]
        < (elbow_position[0] - elbow_singularity_position[0]) * singularity_limit_coeff
        + elbow_singularity_position[2]
        - singularity_offset
    )
    # print(f"elbow_position[2] = {elbow_position[2]}")
    # print(f"elbow_position[0] = {elbow_position[0]}")
    # print(f" < {(elbow_position[0]- elbow_singularity_position[0]) * singularity_limit_coeff +
    # elbow_singularity_position[2] - singularity_offset}")
    # else:
    #     is_ok = is_ok and (elbow_position[2] < elbow_singularity_position[2] - singularity_offset)
    #     print(f"elbow_position[2] = {elbow_position[2]}")
    #     print(f"elbow_singularity_position[2] = {elbow_singularity_position[2] - singularity_offset}")
    # print(f" is_ok = {is_ok}")
    return is_ok


def is_valid_angle(angle: float, interval: npt.NDArray[np.float64]) -> bool:
    """Check if an angle is in the interval"""
    if interval[0] % (2 * np.pi) == interval[1] % (2 * np.pi):
        return True
    if interval[0] < interval[1]:
        return bool(interval[0] <= angle) and (angle <= interval[1])
    return bool(interval[0] <= angle) or (angle <= interval[1])


def make_projection_on_plane(
    P_plane: npt.NDArray[np.float64], normal_vector: npt.NDArray[np.float64], point: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    v = point - P_plane
    dist = np.dot(v, normal_vector)
    projected_point = point - dist * normal_vector
    return np.array(projected_point)


def angle_diff(a: float, b: float) -> float:
    """Returns the smallest distance between 2 angles"""
    d = a - b
    d = ((d + math.pi) % (2 * math.pi)) - math.pi
    return d


def allow_multiturn(new_joints: list[float], prev_joints: list[float], name: str) -> list[float]:
    """This function will always guarantee that the joint takes the shortest path to the new position.
    The practical effect is that it will allow the joint to rotate more than 2pi if it is the shortest path.
    """
    new_joints = copy.deepcopy(new_joints)
    for i in range(len(new_joints)):
        # if i == 0:
        #     self.logger.warning(
        #         f"Joint 6: [{new_joints[i]}, {prev_joints[i]}], angle_diff: {angle_diff(new_joints[i], prev_joints[i])}"
        #     )
        diff = angle_diff(new_joints[i], prev_joints[i])
        new_joints[i] = prev_joints[i] + diff
    # Temp : showing a warning if a multiturn is detected. TODO do better. This info is critical
    # and should be saved dyamically on disk.
    # indexes_that_can_multiturn = [0, 2, 6]
    # for index in indexes_that_can_multiturn:
    # if abs(new_joints[index]) > np.pi:
    #     logger.warning(
    #         f" {name} Multiturn detected on joint {index} with value: {new_joints[index]} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
    #         throttle_duration_sec=1,
    #     )
    # TEMP forbidding multiturn
    # new_joints[index] = np.sign(new_joints[index]) * np.pi
    return new_joints


def limit_orbita3d_joints(joints: list[float], orbita3D_max_angle: float) -> list[float]:
    """Casts the 3 orientations to ensure the orientation is reachable by an Orbita3D. i.e. casting into Orbita's cone."""
    # self.logger.info(f"HEAD initial: {joints}")
    joints = copy.deepcopy(joints)
    rotation = R.from_euler("XYZ", [joints[0], joints[1], joints[2]], degrees=False)
    new_joints = rotation.as_euler("ZYZ", degrees=False)
    new_joints[1] = min(orbita3D_max_angle, max(-orbita3D_max_angle, new_joints[1]))
    rotation = R.from_euler("ZYZ", new_joints, degrees=False)
    [roll, pitch, yaw] = rotation.as_euler("XYZ", degrees=False)
    joints = [float(roll), float(pitch), float(yaw)]
    # self.logger.info(f"HEAD final: {new_joints}")
    return joints


def limit_orbita3d_joints_wrist(joints: list[float], orbita3D_max_angle: float) -> list[float]:
    """Casts the 3 orientations to ensure the orientation is reachable by an Orbita3D using the wrist conventions.
    i.e. casting into Orbita's cone."""
    joints = copy.deepcopy(joints)
    wrist_joints = joints[4:7]

    wrist_joints = limit_orbita3d_joints(wrist_joints, orbita3D_max_angle)

    joints[4:7] = wrist_joints

    return joints


def multiturn_safety_check(
    joints: list[float], shoulder_pitch_limit: float, elbow_yaw_limit: float, wrist_yaw_limit: float, state: str
) -> list[float]:
    """Limit the number of turns allowed on the joints"""
    joints = copy.deepcopy(joints)
    # shoulder pitch
    if joints[0] > shoulder_pitch_limit:
        joints[0] = shoulder_pitch_limit
        state += "\n" + "EMERGENCY STOP: shoulder pitch limit reached"
    if joints[0] < -shoulder_pitch_limit:
        joints[0] = -shoulder_pitch_limit
        state += "\n" + "EMERGENCY STOP: shoulder pitch limit reached"
    # elbow yaw
    if joints[2] > elbow_yaw_limit:
        joints[2] = elbow_yaw_limit
        state += "\n" + "EMERGENCY STOP: elbow yaw limit reached"
    if joints[2] < -elbow_yaw_limit:
        joints[2] = -elbow_yaw_limit
        state += "\n" + "EMERGENCY STOP: elbow yaw limit reached"
    # wrist yaw
    if joints[6] > wrist_yaw_limit:
        joints[6] = wrist_yaw_limit
        state += "\n" + "EMERGENCY STOP: wrist yaw limit reached"
    if joints[6] < -wrist_yaw_limit:
        joints[6] = -wrist_yaw_limit
        state += "\n" + "EMERGENCY STOP: wrist yaw limit reached"
    return joints


def continuity_check(joints: list[float], previous_joints: list[float], max_angulare_change: float, state: str) -> list[float]:
    """Check the continuity of the joints"""
    joints = copy.deepcopy(joints)
    discontinuity = False
    for i in range(len(joints)):
        if abs(angle_diff(joints[i], previous_joints[i])) > max_angulare_change:
            discontinuity = True
    if discontinuity:
        joints = previous_joints
        state += "\n EMERGENCY STOP: joints are not continuous"
    return joints


def show_circle(
    ax: Any,
    center: npt.NDArray[np.float64],
    radius: float,
    normal_vector: npt.NDArray[np.float64],
    intervals: npt.NDArray[np.float64],
    color: str,
) -> None:
    """Show a circle in the 3D space"""
    theta = []
    for interval in intervals:
        angle = np.linspace(interval[0], interval[1], 100)
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


def show_frame(
    ax: Any, position: npt.NDArray[np.float64], rotation_matrix: npt.NDArray[np.float64], color: bool = True, alpha: float = 1.0
) -> None:
    """show a frame in the 3D plot"""
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    x = np.dot(rotation_matrix, x)
    y = np.dot(rotation_matrix, y)
    z = np.dot(rotation_matrix, z)
    max_length = max(np.linalg.norm(x), np.linalg.norm(y), np.linalg.norm(z)) / 20
    if color:
        ax.quiver(position[0], position[1], position[2], x[0], x[1], x[2], color="r", length=max_length, alpha=alpha)
        ax.quiver(position[0], position[1], position[2], y[0], y[1], y[2], color="g", length=max_length, alpha=alpha)
        ax.quiver(position[0], position[1], position[2], z[0], z[1], z[2], color="b", length=max_length, alpha=alpha)
    else:
        ax.quiver(position[0], position[1], position[2], x[0], x[1], x[2], color="black", length=max_length, alpha=alpha)
        ax.quiver(position[0], position[1], position[2], y[0], y[1], y[2], color="dimgray", length=max_length, alpha=alpha)
        ax.quiver(position[0], position[1], position[2], z[0], z[1], z[2], color="lightgrey", length=max_length, alpha=alpha)


def show_point(ax: Any, point: npt.NDArray[np.float64], color: str) -> None:
    """Show a point in the 3D space"""
    ax.plot(point[0], point[1], point[2], "o", color=color)


def get_ik_parameters_from_urdf(urdf_str: str, arm: list[str]) -> dict[str, Any]:
    urdf_file = StringIO(urdf_str)
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    ik_parameters = {}
    for joint in root.findall("joint"):
        for name in arm:
            if joint.attrib["name"] == f"{name}_shoulder_base_joint":
                origin = joint.find("origin").attrib  # type: ignore
                ik_parameters[f"{name}_shoulder_position"] = parse_vector(origin["xyz"])
                orientation = parse_vector(origin["rpy"])
                if name == "r":
                    orientation[0] -= np.pi / 2
                else:
                    orientation[0] += np.pi / 2
                ik_parameters[f"{name}_shoulder_orientation"] = np.degrees(orientation)
            elif joint.attrib["name"] == f"{name}_elbow_base_joint":
                origin = joint.find("origin").attrib  # type: ignore
                position = parse_vector(origin["xyz"])
                ik_parameters[f"{name}_upper_arm_size"] = position[2]
                ik_parameters[f"{name}_elbow_roll_offset"] = -position[0]
            elif joint.attrib["name"] == f"{name}_wrist_base_joint":
                origin = joint.find("origin").attrib  # type: ignore
                position = parse_vector(origin["xyz"])
                ik_parameters[f"{name}_forearm_size"] = position[2]
                ik_parameters[f"{name}_wrist_pitch_offset"] = -position[1]
            elif joint.attrib["name"] == f"{name}_tip_joint":
                origin = joint.find("origin").attrib  # type: ignore
                ik_parameters[f"{name}_tip_position"] = parse_vector(origin["xyz"])
    return ik_parameters


def parse_vector(vector_str: str) -> npt.NDArray[np.float64]:
    return np.array(list(map(float, vector_str.split())))
