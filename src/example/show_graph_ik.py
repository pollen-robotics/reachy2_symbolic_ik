import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import (
    make_homogenous_matrix_from_rotation_matrix,
    rotation_matrix_from_vector,
    show_circle,
    show_frame,
    show_point,
    show_sphere,
)


def show_graph(symbolic_ik: SymbolicIK, goal_pose: npt.NDArray[np.float64]) -> None:
    result = symbolic_ik.is_reachable(goal_pose)
    if result[0]:
        if result[1][0] > result[1][1]:
            theta_middle = (result[1][0] + result[1][1]) / 2 - np.pi
        else:
            theta_middle = (result[1][0] + result[1][1]) / 2
        joints, elbow_position = result[2](theta_middle)
    # SymbolicIK

    intersection_circle = symbolic_ik.get_intersection_circle(goal_pose)
    print(intersection_circle)
    limitation_wrist_circle = symbolic_ik.get_limitation_wrist_circle(goal_pose)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.axes.set_xlim3d(left=-0.2, right=0.6)
    ax.axes.set_ylim3d(bottom=-0.6, top=0.2)
    ax.axes.set_zlim3d(bottom=-0.4, top=0.4)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show(block=False)
    show_point(ax, goal_pose[0], "g")
    rotation_matrix = R.from_euler("xyz", goal_pose[1]).as_matrix()
    grasp_frame = R.from_euler("xyz", [0, np.pi / 2, 0]).as_matrix()
    rotation_matrix = np.dot(rotation_matrix, grasp_frame)
    show_frame(ax, goal_pose[0], rotation_matrix)
    show_point(ax, symbolic_ik.shoulder_position, "b")
    show_point(ax, symbolic_ik.torso_pose, "y")

    plt.draw()
    plt.pause(1)
    show_point(ax, symbolic_ik.wrist_position, "r")
    plt.draw()
    plt.pause(2)
    show_sphere(ax, symbolic_ik.wrist_position, symbolic_ik.forearm_size, "r")
    show_sphere(ax, symbolic_ik.shoulder_position, symbolic_ik.upper_arm_size, "b")
    plt.draw()
    plt.pause(2)
    show_circle(
        ax,
        intersection_circle[0],
        intersection_circle[1],
        intersection_circle[2],
        np.array([[0, 2 * np.pi]]),
        "g",
    )
    plt.draw()
    plt.pause(2)
    show_circle(
        ax,
        limitation_wrist_circle[0],
        limitation_wrist_circle[1],
        limitation_wrist_circle[2],
        np.array([[0, 2 * np.pi]]),
        "y",
    )
    plt.draw()
    plt.pause(2)
    R_torso_intersection = rotation_matrix_from_vector(intersection_circle[2])
    T_torso_intersection = make_homogenous_matrix_from_rotation_matrix(intersection_circle[0], R_torso_intersection)

    point1 = [0, math.cos(result[1][0]) * intersection_circle[1], math.sin(result[1][0]) * intersection_circle[1], 1]
    point2 = [0, math.cos(result[1][1]) * intersection_circle[1], math.sin(result[1][1]) * intersection_circle[1], 1]
    point1 = np.dot(T_torso_intersection, point1)
    point2 = np.dot(T_torso_intersection, point2)
    plt.plot(point1[0], point1[1], point1[2], "ro")
    plt.plot(point2[0], point2[1], point2[2], "ro")
    plt.draw()
    plt.pause(2)

    angle_test = (result[1][0] + result[1][1]) / 2
    print(intersection_circle[1])
    print(angle_test)
    test_point = np.array([0, math.cos(angle_test) * intersection_circle[1], math.sin(angle_test) * intersection_circle[1], 1])
    test_point = np.dot(T_torso_intersection, test_point)
    ax.plot(
        test_point[0],
        test_point[1],
        test_point[2],
        "ro",
    )
    plt.draw()
    plt.pause(2)
    show_circle2(
        ax,
        intersection_circle[0],
        intersection_circle[1],
        intersection_circle[2],
        np.array(result[1]),
        "g",
    )
    # plt.show()
    plt.draw()
    plt.pause(2)
    if result[1][0] > result[1][1]:
        theta_middle = (result[1][0] + result[1][1]) / 2 - np.pi
    else:
        theta_middle = (result[1][0] + result[1][1]) / 2

    elbow_position = symbolic_ik.get_coordinate_cercle(intersection_circle, theta_middle)
    show_point(ax, elbow_position, "b")
    plt.draw()
    plt.pause(2)

    ax.plot(
        [goal_pose[0][0], symbolic_ik.wrist_position[0]],
        [goal_pose[0][1], symbolic_ik.wrist_position[1]],
        [goal_pose[0][2], symbolic_ik.wrist_position[2]],
        "r",
    )
    ax.plot(
        [symbolic_ik.wrist_position[0], elbow_position[0]],
        [symbolic_ik.wrist_position[1], elbow_position[1]],
        [symbolic_ik.wrist_position[2], elbow_position[2]],
        "r",
    )
    ax.plot(
        [elbow_position[0], symbolic_ik.shoulder_position[0]],
        [elbow_position[1], symbolic_ik.shoulder_position[1]],
        [elbow_position[2], symbolic_ik.shoulder_position[2]],
        "r",
    )

    plt.draw()
    plt.pause(2)

    previous_joints = np.array([0, 0, 0, 0, 0, 0, 0])
    # self.elbow_position = self.get_coordinate_circle(self.intersection_circle, theta)
    # goal_orientation = symbolic_ik.goal_pose[1]

    P_torso_shoulder = [symbolic_ik.shoulder_position[0], symbolic_ik.shoulder_position[1], symbolic_ik.shoulder_position[2], 1]
    P_torso_elbow = [symbolic_ik.elbow_position[0], symbolic_ik.elbow_position[1], symbolic_ik.elbow_position[2], 1]
    P_torso_wrist = [symbolic_ik.wrist_position[0], symbolic_ik.wrist_position[1], symbolic_ik.wrist_position[2], 1]
    P_torso_goalPosition = [symbolic_ik.goal_pose[0][0], symbolic_ik.goal_pose[0][1], symbolic_ik.goal_pose[0][2], 1]

    R_torso_shoulder = R.from_euler("xyz", np.radians(symbolic_ik.shoulder_orientation_offset))
    offset_rotation_matrix = R.from_euler("xyz", [0.0, np.pi / 2, 0.0])
    R_torso_shoulder = R_torso_shoulder * offset_rotation_matrix
    R_torso_shoulder = R_torso_shoulder.as_matrix()
    R_shoulder_torso = R_torso_shoulder.T
    P_shoulder_torso = np.dot(-R_shoulder_torso, P_torso_shoulder[:3])
    T_shoulder_torso = make_homogenous_matrix_from_rotation_matrix(P_shoulder_torso, R_shoulder_torso)
    P_shoulder_elbow = np.dot(T_shoulder_torso, P_torso_elbow)

    # Case where the elbow is aligned with the shoulder
    # With current arm configuration this has two impacts:
    # - the shoulder alone is in cinematic singularity -> loose controllability around this point
    # -> in this case the upperarm might rotate quickly even if the elbow displacement is small
    # -> not  this library's responsability
    # - the elbow and the shoulder are aligned -> there is an infinite number of solutions
    # -> this is the library's responsability
    # -> we chose the joints of the previous pose based on the user input in previous_joints

    if P_shoulder_elbow[0] == 0 and P_shoulder_elbow[2] == 0:
        # raise ValueError("Shoulder singularity")
        shoulder_pitch = previous_joints[0]
    else:
        shoulder_pitch = -math.atan2(P_shoulder_elbow[2], P_shoulder_elbow[0])

    R_shoulderPitch_shoulder = R.from_euler("xyz", [0.0, -shoulder_pitch, 0.0]).as_matrix()
    T_shoulderPitch_shoulder = make_homogenous_matrix_from_rotation_matrix(np.array([0.0, 0.0, 0.0]), R_shoulderPitch_shoulder)
    T_shoulderPitch_torso = np.dot(T_shoulderPitch_shoulder, T_shoulder_torso)

    P_shoulderPitch_elbow = np.dot(T_shoulderPitch_torso, P_torso_elbow)

    shoulder_roll = math.atan2(P_shoulderPitch_elbow[1], P_shoulderPitch_elbow[0])

    R_shoulderRoll_shoulderPitch = R.from_euler("xyz", [0.0, 0.0, -shoulder_roll]).as_matrix()
    T_shoulderRoll_shoulderPitch = make_homogenous_matrix_from_rotation_matrix(
        np.array([0.0, 0.0, 0.0]), R_shoulderRoll_shoulderPitch
    )
    T_shoulderRoll_torso = np.dot(T_shoulderRoll_shoulderPitch, T_shoulderPitch_torso)

    T_elbow_torso = T_shoulderRoll_torso
    T_elbow_torso[0][3] -= symbolic_ik.upper_arm_size
    P_elbow_wrist = np.dot(T_elbow_torso, P_torso_wrist)

    # Same as the shoulder singularity but between the wrist and the elbow
    if P_elbow_wrist[1] == 0 and P_elbow_wrist[2] == 0:
        # raise ValueError("Elbow singularity")
        elbow_yaw = previous_joints[2]
    else:
        elbow_yaw = -np.pi / 2 + math.atan2(P_elbow_wrist[2], -P_elbow_wrist[1])
    # if elbow_yaw < -np.pi:
    #     elbow_yaw = elbow_yaw + 2 * np.pi

    R_elbowYaw_elbow = R.from_euler("xyz", np.array([elbow_yaw, 0.0, 0.0])).as_matrix()
    T_elbowYaw_elbow = make_homogenous_matrix_from_rotation_matrix(np.array([0.0, 0.0, 0.0]), R_elbowYaw_elbow)
    T_elbowYaw_torso = np.dot(T_elbowYaw_elbow, T_elbow_torso)

    P_elbowYaw_wrist = np.dot(T_elbowYaw_torso, P_torso_wrist)

    # TODO cas qui arrive probablement en meme temps que la singulartié du coude
    # -> dans ce cas on veut que elbowpitch = 0 -> à verifier
    elbow_pitch = -math.atan2(P_elbowYaw_wrist[2], P_elbowYaw_wrist[0])

    R_elbowPitch_elbowYaw = R.from_euler("xyz", [0.0, -elbow_pitch, 0.0]).as_matrix()
    T_elbowPitch_elbowYaw = make_homogenous_matrix_from_rotation_matrix(np.array([0.0, 0.0, 0.0]), R_elbowPitch_elbowYaw)
    T_elbowPitch_torso = np.dot(T_elbowPitch_elbowYaw, T_elbowYaw_torso)

    T_wrist_torso = T_elbowPitch_torso
    T_wrist_torso[0][3] -= symbolic_ik.forearm_size

    P_wrist_tip = np.dot(T_wrist_torso, P_torso_goalPosition)

    wrist_roll = np.pi - math.atan2(P_wrist_tip[1], -P_wrist_tip[0])
    if wrist_roll > np.pi:
        wrist_roll = wrist_roll - 2 * np.pi

    R_wristRoll_wrist = R.from_euler("xyz", [0.0, 0.0, -wrist_roll]).as_matrix()
    T_wristRoll_wrist = make_homogenous_matrix_from_rotation_matrix(np.array([0.0, 0.0, 0.0]), R_wristRoll_wrist)
    T_wristRol_torso = np.dot(T_wristRoll_wrist, T_wrist_torso)

    P_wristRoll_tip = np.dot(T_wristRol_torso, P_torso_goalPosition)

    wrist_pitch = math.atan2(P_wristRoll_tip[2], P_wristRoll_tip[0])

    R_wristPitch_wrist_Roll = R.from_euler("xyz", [0.0, wrist_pitch, 0.0]).as_matrix()
    T_wristPitch_wrist_Roll = make_homogenous_matrix_from_rotation_matrix(np.array([0.0, 0.0, 0.0]), R_wristPitch_wrist_Roll)
    T_wristPitch_torso = np.dot(T_wristPitch_wrist_Roll, T_wristRol_torso)

    T_tip_torso = T_wristPitch_torso
    T_tip_torso[0][3] -= symbolic_ik.gripper_size

    # elbow_yaw -= np.radians(symbolic_ik.elbow_orientation_offset[2])

    ####

    ax.clear()
    show_point(ax, symbolic_ik.torso_pose, "y")
    ax.axes.set_xlim3d(left=-0.2, right=0.6)
    ax.axes.set_ylim3d(bottom=-0.6, top=0.2)
    ax.axes.set_zlim3d(bottom=-0.4, top=0.4)
    show_point(ax, symbolic_ik.wrist_position, "r")
    show_point(ax, symbolic_ik.shoulder_position, "b")
    show_point(ax, symbolic_ik.elbow_position, "g")
    show_point(ax, goal_pose[0], "g")
    R_grasp_goal_pose = R.from_euler("xyz", goal_pose[1]).as_matrix()
    R_torso_grasp = np.dot(R_grasp_goal_pose, R.from_euler("xyz", [0, np.pi / 2, 0]).as_matrix())
    show_frame(ax, goal_pose[0], R_torso_grasp, alpha=0.5)
    # plt.draw()
    # plt.pause(1)
    ax.plot(
        [goal_pose[0][0], symbolic_ik.wrist_position[0]],
        [goal_pose[0][1], symbolic_ik.wrist_position[1]],
        [goal_pose[0][2], symbolic_ik.wrist_position[2]],
        "r",
    )
    ax.plot(
        [symbolic_ik.wrist_position[0], symbolic_ik.elbow_position[0]],
        [symbolic_ik.wrist_position[1], symbolic_ik.elbow_position[1]],
        [symbolic_ik.wrist_position[2], symbolic_ik.elbow_position[2]],
        "r",
    )
    ax.plot(
        [symbolic_ik.elbow_position[0], symbolic_ik.shoulder_position[0]],
        [symbolic_ik.elbow_position[1], symbolic_ik.shoulder_position[1]],
        [symbolic_ik.elbow_position[2], symbolic_ik.shoulder_position[2]],
        "r",
    )

    plt.draw()
    plt.pause(2)
    show_frame(ax, symbolic_ik.torso_pose, R.from_euler("xyz", [0, 0, 0]).as_matrix())
    plt.draw()
    plt.pause(2)
    show_frame(
        ax, symbolic_ik.shoulder_position, R.from_euler("xyz", np.radians(symbolic_ik.shoulder_orientation_offset)).as_matrix()
    )
    plt.draw()
    plt.pause(2)
    show_frame(
        ax,
        symbolic_ik.shoulder_position,
        R.from_euler("xyz", np.radians(symbolic_ik.shoulder_orientation_offset)).as_matrix(),
        color=False,
    )
    show_frame(ax, symbolic_ik.shoulder_position, R_torso_shoulder)

    plt.draw()
    plt.pause(2)
    show_frame(ax, symbolic_ik.shoulder_position, R_torso_shoulder, color=False)
    R_torso_shoulderPitch = np.dot(R_shoulderPitch_shoulder, R_shoulder_torso).T
    show_frame(ax, symbolic_ik.shoulder_position, R_torso_shoulderPitch)
    plt.draw()
    plt.pause(2)
    show_frame(ax, symbolic_ik.shoulder_position, R_torso_shoulderPitch, color=False)
    R_torso_shoulderRoll = np.dot(R_torso_shoulderPitch, R_shoulderRoll_shoulderPitch.T)
    show_frame(ax, symbolic_ik.shoulder_position, R_torso_shoulderRoll)
    plt.draw()
    plt.pause(2)
    show_frame(ax, symbolic_ik.elbow_position, R_torso_shoulderRoll)
    plt.draw()
    plt.pause(2)
    show_frame(ax, symbolic_ik.shoulder_position, R_torso_shoulder, color=False)
    R_torso_shoulderPitch = np.dot(R_shoulderPitch_shoulder, R_shoulder_torso).T
    show_frame(ax, symbolic_ik.shoulder_position, R_torso_shoulderPitch)
    plt.draw()
    plt.pause(2)
    show_frame(ax, symbolic_ik.elbow_position, R_torso_shoulderRoll, color=False)
    R_torso_elbowYaw = np.dot(R_torso_shoulderRoll, R_elbowYaw_elbow.T)
    show_frame(ax, symbolic_ik.elbow_position, R_torso_elbowYaw)
    plt.draw()
    plt.pause(2)
    show_frame(ax, symbolic_ik.elbow_position, R_torso_elbowYaw, color=False)
    R_torso_elbowPitch = np.dot(R_torso_elbowYaw, R_elbowPitch_elbowYaw.T)
    show_frame(ax, symbolic_ik.elbow_position, R_torso_elbowPitch)
    plt.draw()
    plt.pause(2)
    show_frame(ax, symbolic_ik.elbow_position, R_torso_elbowPitch, color=False)
    show_frame(ax, symbolic_ik.wrist_position, R_torso_elbowPitch)
    plt.draw()
    plt.pause(2)
    show_frame(ax, symbolic_ik.wrist_position, R_torso_elbowPitch, color=False)
    R_torso_wristRoll = np.dot(R_torso_elbowPitch, R_wristRoll_wrist.T)
    show_frame(ax, symbolic_ik.wrist_position, R_torso_wristRoll)
    plt.draw()
    plt.pause(2)
    show_frame(ax, symbolic_ik.wrist_position, R_torso_wristRoll, color=False)
    R_torso_wristPitch = np.dot(R_torso_wristRoll, R_wristPitch_wrist_Roll.T)
    show_frame(ax, symbolic_ik.wrist_position, R_torso_wristPitch)
    plt.draw()
    plt.pause(2)
    show_frame(ax, symbolic_ik.wrist_position, R_torso_wristPitch, color=False)
    show_frame(ax, goal_pose[0], R_torso_wristPitch)
    plt.draw()
    plt.pause(2)
    show_frame(ax, goal_pose[0], R_torso_wristPitch, color=False)
    show_frame(ax, goal_pose[0], R_torso_grasp)

    plt.show()


def show_circle2(
    ax: Any,
    center: npt.NDArray[np.float64],
    radius: float,
    normal_vector: npt.NDArray[np.float64],
    intervalle: npt.NDArray[np.float64],
    color: str,
) -> None:
    valid_theta = []
    wrong_theta = []
    if intervalle[0] < intervalle[1]:
        valid_theta = np.linspace(intervalle[0], intervalle[1], 100)
        wrong_theta = np.linspace(intervalle[1], intervalle[0] + 2 * np.pi, 100)
    else:
        valid_theta = np.linspace(intervalle[0], intervalle[1] + 2 * np.pi, 100)
        wrong_theta = np.linspace(intervalle[1], intervalle[0], 100)

    y = radius * np.cos(valid_theta)
    z = radius * np.sin(valid_theta)
    x = np.zeros(len(valid_theta))
    y2 = radius * np.cos(wrong_theta)
    z2 = radius * np.sin(wrong_theta)
    x2 = np.zeros(len(wrong_theta))
    Rmat = rotation_matrix_from_vector(np.array(normal_vector))
    Tmat = make_homogenous_matrix_from_rotation_matrix(center, Rmat)
    x3 = np.zeros(len(valid_theta))
    y3 = np.zeros(len(valid_theta))
    z3 = np.zeros(len(valid_theta))
    for k in range(len(valid_theta)):
        p = [x[k], y[k], z[k], 1]
        p2 = np.dot(Tmat, p)
        x3[k] = p2[0]
        y3[k] = p2[1]
        z3[k] = p2[2]
    for k in range(len(wrong_theta)):
        p = [x2[k], y2[k], z2[k], 1]
        p2 = np.dot(Tmat, p)
        x2[k] = p2[0]
        y2[k] = p2[1]
        z2[k] = p2[2]

    ax.plot(center[0], center[1], center[2], "o", color=color)
    ax.plot(x3, y3, z3, color)
    ax.plot(x2, y2, z2, "r")


def main() -> None:
    ik_parameters = {
        "r_shoulder_position": np.array([0.0, -0.2, 0.0]),
        "r_shoulder_orientation": [10, 0, 15],
        "r_upper_arm_size": np.float64(0.28),
        "r_forearm_size": np.float64(0.28),
        "r_tip_position": np.array([-0.0, 0.0, 0.10]),
        "r_elbow_roll_offset": 0.03,
        "r_wrist_pitch_offset": 0.03,
        "l_shoulder_position": np.array([0.0, 0.2, 0.0]),
        "l_shoulder_orientation": [-10, 0, 15],
        "l_upper_arm_size": np.float64(0.28),
        "l_forearm_size": np.float64(0.28),
        "l_tip_position": np.array([-0.0, 0.0, 0.10]),
        "l_elbow_roll_offset": 0.03,
        "l_wrist_pitch_offset": 0.03,
    }
    symbolic_ik = SymbolicIK(
        arm="r_arm",
        ik_parameters=ik_parameters,
    )
    # goal_position = [0.55, -0.3, -0.2]
    goal_position = [0.0001, -0.2, -0.65]
    goal_orientation = [0, 0, 0]
    # goal_orientation = [0, -np.pi / 3, np.pi / 5]
    goal_pose = np.array([goal_position, goal_orientation])
    show_graph(symbolic_ik, goal_pose)


if __name__ == "__main__":
    main()
    plt.show()
