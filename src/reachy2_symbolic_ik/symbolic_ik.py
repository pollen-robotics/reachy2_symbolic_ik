import copy
import math
from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.utils import (
    get_singularity_position,
    make_homogenous_matrix_from_rotation_matrix,
    make_projection_on_plane,
    rotation_matrix_from_vector,
    show_circle,
    show_point,
    show_sphere,
)

SHOW_GRAPH = False

if SHOW_GRAPH:
    import matplotlib.pyplot as plt


class SymbolicIK:
    def __init__(
        self,
        arm: str = "r_arm",
        ik_parameters: dict[str, Any] = {},
        elbow_limit: int = 127,
        wrist_limit: np.float64 = np.float64(42.5),
        projection_margin: float = 1e-8,
        backward_limit: float = 0.02,
        normal_vector_margin: float = 1e-7,
        singularity_offset: float = 0.03,
        singularity_limit_coeff: float = 1.0,
    ) -> None:
        if ik_parameters == {}:
            print("Using default parameters")
            ik_parameters = {
                "r_shoulder_position": np.array([0.0, -0.2, 0.0]),
                "r_shoulder_orientation": [-15, 0, 10],
                "r_upper_arm_size": np.float64(0.28),
                "r_forearm_size": np.float64(0.28),
                "r_tip_position": np.array([-0.0, 0.0, 0.10]),
                "l_shoulder_position": np.array([0.0, 0.2, 0.0]),
                "l_shoulder_orientation": [15, 0, -10],
                "l_upper_arm_size": np.float64(0.28),
                "l_forearm_size": np.float64(0.28),
                "l_tip_position": np.array([-0.0, 0.0, 0.10]),
            }

        if arm not in ["r_arm", "l_arm"]:
            raise ValueError("arm should be either 'r_arm' or 'l_arm'")
        self.arm = arm

        for name in ["r", "l"]:
            if self.arm == f"{name}_arm":
                self.shoulder_position = ik_parameters[f"{name}_shoulder_position"]
                self.shoulder_orientation_offset = ik_parameters[f"{name}_shoulder_orientation"]
                self.upper_arm_size = ik_parameters[f"{name}_upper_arm_size"]
                self.forearm_size = ik_parameters[f"{name}_forearm_size"]
                self.tip_position = ik_parameters[f"{name}_tip_position"]
                self.gripper_size = np.linalg.norm(self.tip_position)
                self.max_arm_length = self.upper_arm_size + self.forearm_size + self.gripper_size

        self.torso_pose = np.array([0.0, 0.0, 0.0])

        self.projection_margin = projection_margin
        self.normal_vector_margin = normal_vector_margin
        self.backward_limit = backward_limit
        self.elbow_limit = elbow_limit
        self.shoulder_wrist_min_distance = np.sqrt(
            self.upper_arm_size**2
            + self.forearm_size**2
            - 2 * self.upper_arm_size * self.forearm_size * np.cos(np.radians(180 - self.elbow_limit))
        )
        self.wrist_limit = wrist_limit
        self.singularity_offset = singularity_offset
        self.singularity_limit_coeff = singularity_limit_coeff
        self.elbow_singularity_position, self.wrist_singularity_position = get_singularity_position(
            self.arm, self.shoulder_position, self.shoulder_orientation_offset, self.upper_arm_size, self.forearm_size
        )

    def is_reachable_no_limits(self, goal_pose: npt.NDArray[np.float64]) -> Tuple[bool, npt.NDArray[np.float64], Optional[Any]]:
        """Check if the goal pose is reachable without taking into account the limits of the wrist and the elbow
        Should alway return True"""
        # Change goal pose if goal pose is out of reach or with x <= 0
        _, goal_pose, _ = self.is_pose_in_robot_reach(goal_pose)

        self.goal_pose = goal_pose
        self.wrist_position = self.get_wrist_position(goal_pose)
        # experimental
        if self.wrist_position[0] < self.backward_limit:
            diff = self.backward_limit - self.wrist_position[0]
            goal_pose = np.array([goal_pose[0] + np.array([diff, 0, 0]), goal_pose[1]])
            self.wrist_position = self.get_wrist_position(goal_pose)
            self.goal_pose = goal_pose

        # Check if the wrist is in the arm range and reduce the goal pose if not
        d_shoulder_wrist = np.linalg.norm(self.wrist_position - self.shoulder_position)
        if d_shoulder_wrist > self.upper_arm_size + self.forearm_size:
            self.goal_pose = self.reduce_goal_pose_no_limits(
                goal_pose, d_shoulder_wrist, self.upper_arm_size + self.forearm_size
            )

        if d_shoulder_wrist < self.shoulder_wrist_min_distance:
            goal_pose = self.reduce_goal_pose_no_limits(
                goal_pose, d_shoulder_wrist, np.float64(self.shoulder_wrist_min_distance)
            )
            self.wrist_position = self.get_wrist_position(goal_pose)
            self.goal_pose = goal_pose
        # Get the intersection circle -> with the previous condition we should always find one
        intersection_circle = self.get_intersection_circle(goal_pose)
        if intersection_circle is not None:
            self.intersection_circle = intersection_circle
            return True, np.array([-np.pi, np.pi]), self.get_joints
        else:
            return False, np.array([]), None

    def is_reachable(  # noqa C901
        self, goal_pose: npt.NDArray[np.float64]
    ) -> Tuple[bool, npt.NDArray[np.float64], Optional[Any], str]:
        # print(f" goal pose debut : {goal_pose}")
        """Check if the goal pose is reachable taking into account the limits of the wrist and the elbow"""
        state = ""
        # Change goal pose if goal pose is out of reach or with x <= 0
        is_reachable, goal_pose, reach_state = self.is_pose_in_robot_reach(goal_pose)
        # print(f" __ is_reachable: {is_reachable}, goal_pose: {goal_pose}, reach_state: {reach_state}")
        if not is_reachable:
            state = reach_state
            return False, np.array([]), None, state

        if SHOW_GRAPH:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection="3d")
            self.ax.axes.set_xlim3d(left=-0.4, right=0.4)
            self.ax.axes.set_ylim3d(bottom=-0.4, top=0.4)
            self.ax.axes.set_zlim3d(bottom=-0.4, top=0.4)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
        self.goal_pose = goal_pose
        self.wrist_position = self.get_wrist_position(goal_pose)

        if self.wrist_position[0] < self.backward_limit:
            diff = self.backward_limit - self.wrist_position[0]
            # print(f'wrists position: {self.wrist_position}')
            goal_pose = np.array([goal_pose[0] + np.array([diff, 0, 0]), goal_pose[1]])
            self.wrist_position = self.wrist_position + np.array([diff, 0, 0])
            # print("wrist position: ", self.wrist_position)

            self.goal_pose = goal_pose
        # print(f" foal gose : {goal_pose}")

        # Test if the wrist is in the arm range
        d_shoulder_wrist = np.linalg.norm(self.wrist_position - self.shoulder_position)
        if d_shoulder_wrist > self.upper_arm_size + self.forearm_size:
            state = "wrist out of range"
            # TODO check Trex arm
            return False, np.array([]), None, state

        # Test if the elbow is too much bent

        # experimental
        if d_shoulder_wrist < self.shoulder_wrist_min_distance:
            goal_pose = self.reduce_goal_pose_no_limits(
                goal_pose, d_shoulder_wrist, np.float64(self.shoulder_wrist_min_distance)
            )
            self.wrist_position = self.get_wrist_position(goal_pose)
            self.goal_pose = goal_pose

        # print(f" goal pose __: {goal_pose}")
        # to_asin1 = d_shoulder_wrist / (2 * self.upper_arm_size)
        # to_asin2 = d_shoulder_wrist / (2 * self.forearm_size)
        # alpha = np.arcsin(to_asin1) + np.arcsin(to_asin2) - np.pi
        # if alpha < np.radians(-self.elbow_limit) or alpha > np.radians(self.elbow_limit):
        #     state = "limited by elbow"
        #     return False, np.array([]), None, state

        intersection_circle = self.get_intersection_circle(goal_pose)
        limitation_wrist_circle = self.get_limitation_wrist_circle(goal_pose)

        if intersection_circle is not None:
            self.intersection_circle = intersection_circle
            # Check if the two circles are linked and return the interval of the valid angles
            interval = self.are_circles_linked(intersection_circle, limitation_wrist_circle)
            if len(interval) > 0:
                if SHOW_GRAPH:
                    elbow_position = self.get_elbow_position(interval[0])
                    show_point(self.ax, elbow_position, "r")
                    self.ax.plot(
                        [goal_pose[0][0], self.wrist_position[0]],
                        [goal_pose[0][1], self.wrist_position[1]],
                        [goal_pose[0][2], self.wrist_position[2]],
                        "r",
                    )
                    self.ax.plot(
                        [self.wrist_position[0], elbow_position[0]],
                        [self.wrist_position[1], elbow_position[1]],
                        [self.wrist_position[2], elbow_position[2]],
                        "r",
                    )
                    self.ax.plot(
                        [elbow_position[0], self.shoulder_position[0]],
                        [elbow_position[1], self.shoulder_position[1]],
                        [elbow_position[2], self.shoulder_position[2]],
                        "r",
                    )
                    show_point(self.ax, goal_pose[0], "g")
                    show_point(self.ax, self.wrist_position, "r")
                    show_point(self.ax, self.shoulder_position, "b")
                    show_point(self.ax, self.torso_pose, "y")
                    show_sphere(self.ax, self.wrist_position, self.forearm_size, "r")
                    show_sphere(self.ax, self.shoulder_position, self.upper_arm_size, "b")
                    show_circle(
                        self.ax,
                        intersection_circle[0],
                        intersection_circle[1],
                        intersection_circle[2],
                        np.array([[0, 2 * np.pi]]),
                        "g",
                    )
                    show_circle(
                        self.ax,
                        limitation_wrist_circle[0],
                        limitation_wrist_circle[1],
                        limitation_wrist_circle[2],
                        np.array([[0, 2 * np.pi]]),
                        "y",
                    )
                    plt.show()
                if state == "":
                    state = "reachable"
                return True, interval, self.get_joints, state

            if SHOW_GRAPH:
                show_point(self.ax, goal_pose[0], "g")
                show_point(self.ax, self.wrist_position, "r")
                show_point(self.ax, self.shoulder_position, "b")
                show_point(self.ax, self.torso_pose, "y")
                show_sphere(self.ax, self.wrist_position, self.forearm_size, "r")
                show_sphere(self.ax, self.shoulder_position, self.upper_arm_size, "b")
                show_circle(
                    self.ax,
                    intersection_circle[0],
                    intersection_circle[1],
                    intersection_circle[2],
                    np.array([[0, 2 * np.pi]]),
                    "g",
                )
                show_circle(
                    self.ax,
                    limitation_wrist_circle[0],
                    limitation_wrist_circle[1],
                    limitation_wrist_circle[2],
                    np.array([[0, 2 * np.pi]]),
                    "y",
                )
                plt.show()
            if state == "":
                state = "limited by wrist"
            return False, np.array([]), None, state

        if SHOW_GRAPH:
            show_point(self.ax, goal_pose[0], "g")
            show_point(self.ax, self.wrist_position, "r")
            show_point(self.ax, self.shoulder_position, "b")
            show_point(self.ax, self.torso_pose, "y")
            show_sphere(self.ax, self.wrist_position, self.forearm_size, "r")
            show_sphere(self.ax, self.shoulder_position, self.upper_arm_size, "b")
            show_circle(
                self.ax,
                limitation_wrist_circle[0],
                limitation_wrist_circle[1],
                limitation_wrist_circle[2],
                np.array([[0, 2 * np.pi]]),
                "y",
            )
            plt.show()
        state = "out of reach - should not happen"
        return False, np.array([]), None, state

    def is_pose_in_robot_reach(self, goal_pose: npt.NDArray[np.float64]) -> Tuple[bool, npt.NDArray[np.float64], str]:
        """Reduce the goal pose if the goal pose is out of reach and prevent the tip to go backward"""
        goal_pose = copy.deepcopy(goal_pose)

        goal_position = goal_pose[0]
        d_shoulder_goal = np.linalg.norm(goal_pose[0] - self.shoulder_position)
        state = ""
        is_reachable = True
        # Reduce the goal pose if the goal pose is out of reach
        if d_shoulder_goal > self.max_arm_length:
            is_reachable = False
            # Make projection of the goal position on the reachable sphere
            goal_position = goal_pose[0]
            direction = goal_position - self.shoulder_position
            direction = direction / (np.linalg.norm(direction) + self.projection_margin)
            goal_position = self.shoulder_position + direction * self.max_arm_length
            state = "Pose out of reach"

        # Avoid the tip to go backward
        if goal_position[0] < self.backward_limit:
            is_reachable = False
            goal_position[0] = self.backward_limit
            state = "Backward pose"
        return is_reachable, np.array([goal_position, goal_pose[1]]), state

    def limit_wrist_pose(
        self, goal_pose: npt.NDArray[np.float64], wrist_position: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Limit the wrist pose if the wrist is out of reach"""

        # TODO : use offset
        M_torso_shoulderYaw = R.from_euler("xyz", [0, 0, np.radians(self.shoulder_orientation_offset[2])]).as_matrix()
        M_shoulderYaw_torso = M_torso_shoulderYaw.T
        P_torso_shoulder = np.dot(-M_shoulderYaw_torso, self.shoulder_position[:3])
        T_shoulderYaw_torso = make_homogenous_matrix_from_rotation_matrix(P_torso_shoulder, M_shoulderYaw_torso)
        P_shoulderYaw_wrist = np.dot(
            T_shoulderYaw_torso, np.array([wrist_position[0], wrist_position[1], wrist_position[2], 1])
        )

        if P_shoulderYaw_wrist[0] > 0:
            diff = P_shoulderYaw_wrist[2] - (
                P_shoulderYaw_wrist[0] * self.singularity_limit_coeff
                + self.wrist_singularity_position[2]
                - self.singularity_offset
            )
        else:
            diff = P_shoulderYaw_wrist[2] - (self.wrist_singularity_position[2] - self.singularity_offset)
        if diff > 0:
            new_goal_position = goal_pose[0] - np.array([0, 0, diff])
            new_wrist_position = wrist_position - np.array([0, 0, diff])
            return np.array([new_goal_position, goal_pose[1]]), new_wrist_position
        return goal_pose, wrist_position

    def reduce_goal_pose_no_limits(
        self, pose: npt.NDArray[np.float64], d_shoulder_wrist: np.float64, d_shoulder_wrist_max: np.float64
    ) -> npt.NDArray[np.float64]:
        """Reduce the goal pose if the wrist is out of reach"""
        # Make projection of the wrist position on the reachable sphere of the wrist
        # and apply the same projection to the goal position
        direction = self.wrist_position - self.shoulder_position
        direction = direction / (np.linalg.norm(d_shoulder_wrist) + self.projection_margin)
        new_wrist_position = self.shoulder_position + direction * d_shoulder_wrist_max
        diff_wrist = new_wrist_position - self.wrist_position
        goal_position = pose[0] + diff_wrist
        self.wrist_position = new_wrist_position
        return np.array([goal_position, pose[1]])

    # def reduce_goal_pose_no_limits_min(
    #     self, pose: npt.NDArray[np.float64], d_shoulder_wrist: np.float64, d_shoulder_wrist_min: np.float64
    # ) -> npt.NDArray[np.float64]:
    #     """Reduce the goal pose if the wrist is out of reach"""
    #     # Make projection of the wrist position on the reachable sphere of the wrist
    #     # and apply the same projection to the goal position
    #     direction = self.wrist_position - self.shoulder_position
    #     direction = direction / (np.linalg.norm(d_shoulder_wrist) + self.projection_margin)
    #     new_wrist_position = self.shoulder_position + direction * d_shoulder_wrist_min
    #     diff_wrist = new_wrist_position - self.wrist_position
    #     goal_position = pose[0] + diff_wrist
    #     self.wrist_position = new_wrist_position
    #     # print(f"new_wrist_position: {new_wrist_position}")
    #     return np.array([goal_position, pose[1]])

    def get_intersection_circle(
        self, goal_pose: npt.NDArray[np.float64]
    ) -> Optional[Tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]]]:
        """Get the intersection circle between the shoulder sphere and the wrist sphere"""
        P_shoulder_wrist = self.wrist_position - self.shoulder_position

        # Check if the two spheres are linked
        d = np.sqrt(P_shoulder_wrist[0] ** 2 + P_shoulder_wrist[1] ** 2 + P_shoulder_wrist[2] ** 2)
        if d > self.upper_arm_size + self.forearm_size:
            return None
        # Rotation matrix from the intersection frame (the x axe is the vector between the shoulder and the wrist)
        # to the torso frame
        M_torso_intersection = R.from_euler(
            "xyz",
            [
                0.0,
                -math.asin(P_shoulder_wrist[2] / d),
                math.atan2(P_shoulder_wrist[1], P_shoulder_wrist[0]),
            ],
        )
        # Get the radius of the intersection circle
        radius = (
            1
            / (2 * d)
            * np.sqrt(4 * d**2 * self.upper_arm_size**2 - (d**2 - self.forearm_size**2 + self.upper_arm_size**2) ** 2)
        )
        # Get the center of the intersection circle in the torso frame
        P_intersection_center = np.array([(d**2 - self.forearm_size**2 + self.upper_arm_size**2) / (2 * d), 0, 0])
        P_shoulder_center = M_torso_intersection.apply(P_intersection_center)
        P_torso_center = P_shoulder_center + self.shoulder_position
        # Get the normal vector of the intersection circle in the torso frame
        V_intersection_normal = np.array([1.0, 0.0, 0.0])
        V_torso_normal = M_torso_intersection.apply(V_intersection_normal)
        return P_torso_center, radius, V_torso_normal

    def get_limitation_wrist_circle(
        self, goal_pose: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]]:
        """Get the limitation circle of the wrist"""
        # The normal vector is going out of the wrist sphere
        normal_vector = np.array(
            [
                self.wrist_position[0] - goal_pose[0][0],
                self.wrist_position[1] - goal_pose[0][1],
                self.wrist_position[2] - goal_pose[0][2],
            ]
        )
        radius = np.sin(np.radians(self.wrist_limit)) * self.forearm_size
        vector = normal_vector / np.linalg.norm(normal_vector) * np.sqrt(self.forearm_size**2 - radius**2)
        center = self.wrist_position + vector
        return center, radius, normal_vector

    def get_wrist_position(self, goal_pose: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Get the wrist position from the goal pose"""
        M_torso_goalPosition = R.from_euler("xyz", goal_pose[1]).as_matrix()
        T_torso_goalPosition = make_homogenous_matrix_from_rotation_matrix(goal_pose[0], M_torso_goalPosition)
        P_goalPosition_wrist = np.array([-self.tip_position[0], self.tip_position[1], self.tip_position[2], 1.0])
        P_torso_wrist = np.array(np.dot(T_torso_goalPosition, P_goalPosition_wrist))
        # P_torso_wrist = np.array(np.dot(T_torso_goalPosition, np.array([0.0, 0.0, self.gripper_size, 1.0])))
        return P_torso_wrist[:3]

    def are_circles_linked(
        self,
        intersection_circle: Tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]],
        limitation_wrist_circle: Tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """Get the intersection of the two circles and return the interval of the valid angles"""
        radius1 = limitation_wrist_circle[1]
        radius2 = intersection_circle[1]

        p1 = np.array(
            [
                limitation_wrist_circle[0][0] - self.wrist_position[0],
                limitation_wrist_circle[0][1] - self.wrist_position[1],
                limitation_wrist_circle[0][2] - self.wrist_position[2],
            ]
        )
        p2 = np.array(
            [
                intersection_circle[0][0] - self.wrist_position[0],
                intersection_circle[0][1] - self.wrist_position[1],
                intersection_circle[0][2] - self.wrist_position[2],
            ]
        )

        V_torso_normal1 = np.array(limitation_wrist_circle[2])
        V_torso_normal2 = np.array(intersection_circle[2])

        R_torso_intersection = rotation_matrix_from_vector(V_torso_normal2)
        T_torso_intersection = make_homogenous_matrix_from_rotation_matrix(p2, R_torso_intersection)

        R_intersection_torso = R_torso_intersection.T
        P_intersection_torso = np.dot(-R_intersection_torso, p2)
        T_intersection_torso = make_homogenous_matrix_from_rotation_matrix(P_intersection_torso, R_intersection_torso)

        R_torso_limitation = rotation_matrix_from_vector(V_torso_normal1)
        R_limitation_torso = R_torso_limitation.T
        P_limitation_torso = np.dot(-R_limitation_torso, p1)
        T_limitation_torso = make_homogenous_matrix_from_rotation_matrix(P_limitation_torso, R_limitation_torso)

        P_torso_center2 = np.array([p2[0], p2[1], p2[2], 1])
        P_limitation_intersectionCenter = np.dot(T_limitation_torso, P_torso_center2)

        if np.any(V_torso_normal1 != 0):
            V_torso_normal1 = V_torso_normal1 / np.linalg.norm(V_torso_normal1)
        if np.any(V_torso_normal2 != 0):
            V_torso_normal2 = V_torso_normal2 / np.linalg.norm(V_torso_normal2)

        # Check if the two circles are parallel
        if np.all(np.abs(V_torso_normal2 - V_torso_normal1) < self.normal_vector_margin) or np.all(
            np.abs(V_torso_normal2 + V_torso_normal1) < self.normal_vector_margin
        ):
            # if the two circles are parallel the interval is full if above the wrist limitation circle
            # (that means the intersection is in the autorized part of the wrist sphere) and empty otherwise
            if P_limitation_intersectionCenter[0] > 0:
                return np.array([-np.pi, np.pi])
            else:
                return np.array([])

        else:
            # Find the line of intersection of the planes
            q, v = self.points_of_nearest_approach(p1, V_torso_normal1, p2, V_torso_normal2)
            if len(q) == 0:
                # if the two circles are not parallel and the line of intersection of the planes is empty
                # -> not suppose to happen?
                if P_limitation_intersectionCenter[0] > 0:
                    return np.array([-np.pi, np.pi])
                else:
                    return np.array([])
            # Find the intersection points of the circles with the line of intersection of the planes
            points = self.intersection_circle_line_3d_vd(p1, radius1, v, q)
            if points is None:
                # Happens when the two circles are not linked but not parallel
                # -> Check if the intersection is in the autorized part of the wrist sphere
                if P_limitation_intersectionCenter[0] > 0:
                    return np.array([-np.pi, np.pi])
                else:
                    return np.array([])
            else:
                # Get the right interval of the valid angles from the intersection points
                interval = self.get_interval_from_intersection(
                    points, T_intersection_torso, T_torso_intersection, T_limitation_torso, radius2
                )
                return interval

    def get_interval_from_intersection(
        self,
        points: npt.NDArray[np.float64],
        T_intersection_torso: npt.NDArray[np.float64],
        T_torso_intersection: npt.NDArray[np.float64],
        T_limitation_torso: npt.NDArray[np.float64],
        radius2: float,
    ) -> npt.NDArray[np.float64]:
        """Get the interval of the valid angles from the intersection points"""
        # If there is only one intersection point the interval is the angle of the point
        if len(points) == 1:
            # TODO check the case where the intersection is in the autorized part of the wrist sphere
            point = [points[0][0], points[0][1], points[0][2], 1]
            point_in_sphere_frame = np.dot(T_intersection_torso, point)
            angle = math.atan2(point_in_sphere_frame[2], point_in_sphere_frame[1])
            interval = np.array([angle, angle])
            return interval

        # If there are two intersection points there is two intervals possible, we have to determine which one is valid
        if len(points) == 2:
            point1 = [points[0][0], points[0][1], points[0][2], 1]
            point2 = [points[1][0], points[1][1], points[1][2], 1]
            point1_in_sphere_frame = np.dot(T_intersection_torso, point1)
            self.intersection = (
                point1[0] + self.wrist_position[0],
                point1[1] + self.wrist_position[1],
                point1[2] + self.wrist_position[2],
            )

            point2_in_sphere_frame = np.dot(T_intersection_torso, point2)
            # these angles are the limits of the valid arc circle
            angle1 = math.atan2(point1_in_sphere_frame[2], point1_in_sphere_frame[1])
            angle2 = math.atan2(point2_in_sphere_frame[2], point2_in_sphere_frame[1])

            [angle1, angle2] = sorted([angle1, angle2])
            angle_test = (angle1 + angle2) / 2

            # finding which side of the circle is valid by testing the middle point of the arc
            P_intersection_testPoint = np.array([0, math.cos(angle_test) * radius2, math.sin(angle_test) * radius2, 1])

            # transforming the test point to the torso frame and then to the wrist limitation frame
            P_torso_testPoint = np.dot(T_torso_intersection, P_intersection_testPoint)
            if SHOW_GRAPH:
                self.ax.plot(
                    P_torso_testPoint[0] + self.wrist_position[0],
                    P_torso_testPoint[1] + self.wrist_position[1],
                    P_torso_testPoint[2] + self.wrist_position[2],
                    "ro",
                )
            P_limitation_testPoint = np.dot(T_limitation_torso, P_torso_testPoint)

            # if the test point is in the autorized part of the wrist sphere the interval is valid
            # otherwise the convention is to take the other interval -> interval[0] > interval[1
            if P_limitation_testPoint[0] > 0:
                interval = np.array([angle1, angle2])
            else:
                interval = np.array([angle2, angle1])
        return interval

    def intersection_point(
        self,
        v1: npt.NDArray[np.float64],
        p01: npt.NDArray[np.float64],
        v2: npt.NDArray[np.float64],
        p02: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Find the intersection point of two lines"""
        A = np.vstack((v1, -v2)).T
        b = np.subtract(p02, p01)
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        if np.all(np.isclose(params, params[0])):
            return np.array([])

        intersection = v1 * np.float64(params[0]) + p01
        return intersection

    def points_of_nearest_approach(
        self,
        p1: npt.NDArray[np.float64],
        V_torso_normal1: npt.NDArray[np.float64],
        p2: npt.NDArray[np.float64],
        V_torso_normal2: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Find the line of intersection of the planes containing the circles"""
        # vector perpendicular to both circles
        v = np.cross(V_torso_normal1, V_torso_normal2)
        v = v / np.linalg.norm(v)

        # Vectors normal of the plans containing the circles
        vect1 = np.cross(v, V_torso_normal1)
        vect2 = np.cross(v, V_torso_normal2)

        # Find the intersection point of the two lines defined by the normal vectors of the circles and their center
        q = np.array(self.intersection_point(vect1, p1, vect2, p2))
        return q, v

    def intersection_circle_line_3d_vd(
        self,
        center: npt.NDArray[np.float64],
        radius: float,
        direction: npt.NDArray[np.float64],
        point_on_line: npt.NDArray[np.float64],
    ) -> Optional[npt.NDArray[np.float64]]:
        """Find the intersection points of a circle and a line"""
        a = np.dot(direction, direction)
        b = 2 * np.dot(direction, np.subtract(point_on_line, center))
        c = np.dot(np.subtract(point_on_line, center), np.subtract(point_on_line, center)) - radius**2
        discriminant = b**2 - 4 * a * c

        # No intersection
        if discriminant < 0:
            return None
        # One intersection
        elif discriminant == 0:
            t = -b / (2 * a)
            intersection = point_on_line + t * direction
            return np.array([intersection])
        # Two intersections
        else:
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)
            intersection1 = point_on_line + t1 * direction
            intersection2 = point_on_line + t2 * direction

        points = np.vstack((intersection1, intersection2))
        if SHOW_GRAPH:
            for point in points:
                plt.plot(
                    point[0] + self.wrist_position[0],
                    point[1] + self.wrist_position[1],
                    point[2] + self.wrist_position[2],
                    "ro",
                )
        return points

    def make_elbow_projection(
        self,
        goal_pose: npt.NDArray[np.float64],
        elbow_position: npt.NDArray[np.float64],
        singularity_limit_coeff: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        alpha = np.arctan2(-singularity_limit_coeff, 1)
        M_limits = R.from_euler("xyz", [0, alpha, 0]).as_matrix()

        T_limits = make_homogenous_matrix_from_rotation_matrix(self.elbow_singularity_position, M_limits)
        P_limits = np.dot(T_limits, np.array([0, 0, -self.singularity_offset, 1]))

        T_limits = make_homogenous_matrix_from_rotation_matrix(P_limits, M_limits)

        # get normal vector
        n1 = np.array([1, 0, 0, 1])
        n2 = np.array([0, 1, 0, 1])
        n1 = np.dot(T_limits, n1)
        n2 = np.dot(T_limits, n2)
        v1 = n1 - P_limits
        v2 = n2 - P_limits
        v3 = np.cross(v1[:3], v2[:3])
        v3 = v3 / np.linalg.norm(v3)

        projected_center = make_projection_on_plane(P_limits[:3], v3, self.shoulder_position)
        radius = np.sqrt(self.upper_arm_size**2 - np.linalg.norm(self.shoulder_position - projected_center) ** 2)

        projected_elbow = make_projection_on_plane(P_limits[:3], v3, elbow_position)
        V_center_projection = projected_elbow - projected_center
        new_elbow_position = projected_center + radius * (V_center_projection / np.linalg.norm(V_center_projection))

        diff = new_elbow_position - elbow_position
        new_goal_position = goal_pose[0] + diff
        # print(f"new_goal_position: {new_goal_position}")

        return np.array([new_goal_position, goal_pose[1]]), new_elbow_position

    def get_elbow_position(self, theta: float) -> npt.NDArray[np.float64]:
        """Get the position of the elbow from the intersection circle and the angle theta"""
        R_torso_intersection = rotation_matrix_from_vector(np.array(self.intersection_circle[2]))
        T_torso_intersection = make_homogenous_matrix_from_rotation_matrix(self.intersection_circle[0], R_torso_intersection)
        # Get the point on the circle in the intersection frame
        x = 0
        y = self.intersection_circle[1] * np.cos(theta)
        z = self.intersection_circle[1] * np.sin(theta)
        P_intersection_point = np.array([x, y, z, 1])
        # Get the point on the circle in the torso frame
        P_torso_point = np.array(np.dot(T_torso_intersection, P_intersection_point))
        return P_torso_point

    def get_joints(
        self, theta: float, previous_joints: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get the joints from the angle theta
        The previous joints is used to avoid the singularity of the elbow and the shoulder
        Return the joints cast between -pi and pi
        """
        # print(f"goal pose: {self.goal_pose}")
        # Get the position of the elbow from theta
        self.elbow_position = self.get_elbow_position(theta)

        if (
            self.elbow_position[2]
            > (self.elbow_position[0] - self.elbow_singularity_position[0]) * self.singularity_limit_coeff
            + self.elbow_singularity_position[2]
            - self.singularity_offset
        ):
            self.goal_pose, self.elbow_position = self.make_elbow_projection(
                self.goal_pose, self.elbow_position[:3], self.singularity_limit_coeff
            )
            # print("_____elbow projection______")
            self.wrist_position = self.get_wrist_position(self.goal_pose)
        # print(f"post elbow projection: {self.goal_pose}")
        goal_orientation = self.goal_pose[1]

        P_torso_shoulder = [self.shoulder_position[0], self.shoulder_position[1], self.shoulder_position[2], 1]
        P_torso_elbow = [self.elbow_position[0], self.elbow_position[1], self.elbow_position[2], 1]
        P_torso_wrist = [self.wrist_position[0], self.wrist_position[1], self.wrist_position[2], 1]
        P_torso_goalPosition = [self.goal_pose[0][0], self.goal_pose[0][1], self.goal_pose[0][2], 1]

        # Get the shoulder frame
        M_torso_shoulder = R.from_euler("xyz", np.radians(self.shoulder_orientation_offset))

        # Add a offset to the shoulder orientation because the goal pose is in the grasp frame
        offset_rotation_matrix = R.from_euler("xyz", [0.0, np.pi / 2, 0.0])
        M_torso_shoulder = M_torso_shoulder * offset_rotation_matrix
        M_torso_shoulder = M_torso_shoulder.as_matrix()

        # The shoulder fram is now the torso frame with the shoulder orientation and the offset of the grasp frame
        M_shoulder_torso = M_torso_shoulder.T
        P_shoulder_torso = np.dot(-M_shoulder_torso, P_torso_shoulder[:3])
        T_shoulder_torso = make_homogenous_matrix_from_rotation_matrix(P_shoulder_torso, M_shoulder_torso)

        # The elbow position in the shoulder frame is used to find the shoulder pitch joint
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

        # ShoulderPitch frame is the shoulder frame with the shoulder pitch rotation
        M_shoulderPitch_shoulder = R.from_euler("xyz", [0.0, -shoulder_pitch, 0.0]).as_matrix()
        T_shoulderPitch_shoulder = make_homogenous_matrix_from_rotation_matrix(
            np.array([0.0, 0.0, 0.0]), M_shoulderPitch_shoulder
        )
        T_shoulderPitch_torso = np.dot(T_shoulderPitch_shoulder, T_shoulder_torso)

        # The elbow position in the shoulderPitch frame is used to find the shoulder roll joint
        P_shoulderPitch_elbow = np.dot(T_shoulderPitch_torso, P_torso_elbow)
        shoulder_roll = math.atan2(P_shoulderPitch_elbow[1], P_shoulderPitch_elbow[0])

        # The shoulderRoll frame is the shoulderPitch frame with the shoulder roll rotation
        M_shoulderRoll_shoulderPitch = R.from_euler("xyz", [0.0, 0.0, -shoulder_roll]).as_matrix()
        T_shoulderRoll_shoulderPitch = make_homogenous_matrix_from_rotation_matrix(
            np.array([0.0, 0.0, 0.0]), M_shoulderRoll_shoulderPitch
        )
        T_shoulderRoll_torso = np.dot(T_shoulderRoll_shoulderPitch, T_shoulderPitch_torso)

        # The elbow frame is the shoulderRoll frame with the elbow position
        T_elbow_torso = T_shoulderRoll_torso
        T_elbow_torso[0][3] -= self.upper_arm_size

        # The wrist position in the elbow frame is used to find the elbow yaw joint
        P_elbow_wrist = np.dot(T_elbow_torso, P_torso_wrist)
        # Same as the shoulder singularity but between the wrist and the elbow
        if P_elbow_wrist[1] == 0 and P_elbow_wrist[2] == 0:
            # raise ValueError("Elbow singularity")
            elbow_yaw = previous_joints[2]
        else:
            elbow_yaw = -np.pi / 2 + math.atan2(P_elbow_wrist[2], -P_elbow_wrist[1])

        # ElbowYaw frame is the elbow frame with the elbow yaw rotation
        M_elbowYaw_elbow = R.from_euler("xyz", np.array([elbow_yaw, 0.0, 0.0])).as_matrix()
        T_elbowYaw_elbow = make_homogenous_matrix_from_rotation_matrix(np.array([0.0, 0.0, 0.0]), M_elbowYaw_elbow)
        T_elbowYaw_torso = np.dot(T_elbowYaw_elbow, T_elbow_torso)

        # The wrist position in the elbowYaw frame is used to find the elbow pitch joint
        P_elbowYaw_wrist = np.dot(T_elbowYaw_torso, P_torso_wrist)
        # TODO cas qui arrive probablement en meme temps que la singulartié du coude
        # -> dans ce cas on veut que elbowpitch = 0 -> à verifier
        elbow_pitch = -math.atan2(P_elbowYaw_wrist[2], P_elbowYaw_wrist[0])

        # ElbowPitch frame is the elbowYaw frame with the elbow pitch rotation
        R_elbowPitch_elbowYaw = R.from_euler("xyz", [0.0, -elbow_pitch, 0.0]).as_matrix()
        T_elbowPitch_elbowYaw = make_homogenous_matrix_from_rotation_matrix(np.array([0.0, 0.0, 0.0]), R_elbowPitch_elbowYaw)
        T_elbowPitch_torso = np.dot(T_elbowPitch_elbowYaw, T_elbowYaw_torso)

        # The wrist frame is the elbowPitch frame with the wrist position
        T_wrist_torso = T_elbowPitch_torso
        T_wrist_torso[0][3] -= self.forearm_size

        M_goal_rotation = R.from_euler("xyz", goal_orientation)
        T_torso_goalPose = make_homogenous_matrix_from_rotation_matrix(P_torso_goalPosition, M_goal_rotation.as_matrix())
        P_goalPose_tip = np.array([-self.tip_position[0], self.tip_position[1], 0, 1.0])
        P_torso_tip = np.dot(T_torso_goalPose, P_goalPose_tip)

        # The goal position in the wrist frame is used to find the wrist roll joint
        P_wrist_tip = np.dot(T_wrist_torso, P_torso_tip)
        wrist_roll = np.pi - math.atan2(P_wrist_tip[1], -P_wrist_tip[0])
        if wrist_roll > np.pi:
            wrist_roll = wrist_roll - 2 * np.pi

        # WristRoll frame is the wrist frame with the wrist roll rotation
        R_wristRoll_wrist = R.from_euler("xyz", [0.0, 0.0, -wrist_roll]).as_matrix()
        T_wristRoll_wrist = make_homogenous_matrix_from_rotation_matrix(np.array([0.0, 0.0, 0.0]), R_wristRoll_wrist)
        T_wristRol_torso = np.dot(T_wristRoll_wrist, T_wrist_torso)

        # The goal position in the wristRoll frame is used to find the wrist pitch joint
        P_wristRoll_tip = np.dot(T_wristRol_torso, P_torso_tip)
        wrist_pitch = math.atan2(P_wristRoll_tip[2], P_wristRoll_tip[0])

        # WristPitch frame is the wristRoll frame with the wrist pitch rotation
        R_wristPitch_wrist_Roll = R.from_euler("xyz", [0.0, wrist_pitch, 0.0]).as_matrix()
        T_wristPitch_wrist_Roll = make_homogenous_matrix_from_rotation_matrix(
            np.array([0.0, 0.0, 0.0]), R_wristPitch_wrist_Roll
        )
        T_wristPitch_torso = np.dot(T_wristPitch_wrist_Roll, T_wristRol_torso)

        # The tip frame is the wristPitch frame with the goal position
        T_tip_torso = T_wristPitch_torso
        T_tip_torso[0][3] -= self.tip_position[2]

        M_torso_goal = R.from_euler("xyz", goal_orientation)

        # Take a point in the goal frame and find it in the torso frame
        P_goal_point = [0.1, 0.0, 0.0, 1.0]
        T_torso_goal = make_homogenous_matrix_from_rotation_matrix(P_torso_tip, M_torso_goal.as_matrix())
        P_torso_point = np.dot(T_torso_goal, P_goal_point)

        # Use the point in tip frame to find the wrist yaw joint
        P_tip_point = np.dot(T_tip_torso, P_torso_point)
        wrist_yaw = -math.atan2(P_tip_point[1], P_tip_point[2])

        joints = np.array([shoulder_pitch, shoulder_roll, elbow_yaw, elbow_pitch, wrist_roll, -wrist_pitch, -wrist_yaw])
        # if self.arm == "r_arm":
        #     print(f"joints: {list(joints)}")
        elbow_limit = np.radians(self.elbow_limit)
        if joints[3] > elbow_limit:
            # if self.arm == "r_arm":
            #     print(f"joints: {joints}")
            joints[3] = elbow_limit
        if joints[3] < -elbow_limit:
            # if self.arm == "r_arm":
            #     print(f"joints: {joints}")
            joints[3] = -elbow_limit

        return joints, self.elbow_position
