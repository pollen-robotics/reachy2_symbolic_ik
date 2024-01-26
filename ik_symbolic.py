from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
import copy
import tf_transformations
import math
import numpy as np
import matplotlib.pyplot as plt
import time

SHOW_GRAPH = False


class IK_symbolic:
    def __init__(
        self,
        grasp_marker_tip_len=0.2,
        grasp_marker_width=20,
        upper_arm_size=0.28,
        forearm_size=0.28,
        gripper_size=0.15,
        x_offset=3.0,
        wrist_limit=45,
        shoulder_orientation_offset=[10, 0, 15],
    ):
        self.grasp_marker_tip_len = grasp_marker_tip_len
        self.grasp_marker_width = grasp_marker_width
        self.upper_arm_size = upper_arm_size
        self.forearm_size = forearm_size
        self.gripper_size = gripper_size
        self.x_offset = x_offset
        self.wrist_limit = wrist_limit
        self.shoulder_orientation_offset = shoulder_orientation_offset
        self.torso_pose = [0.0, 0.0, 0.0]
        self.shoulder_position = [0.0, -0.2, 0.0]

        if SHOW_GRAPH:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection="3d")
            self.ax.axes.set_xlim3d(left=-0.4, right=0.4)
            self.ax.axes.set_ylim3d(bottom=-0.4, top=0.4)
            self.ax.axes.set_zlim3d(bottom=-0.4, top=0.4)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")

    def is_reachable(self, goal_pose):
        self.goal_pose = goal_pose
        self.wrist_position = self.get_wrist_position(goal_pose)
        limitation_wrist_circle = self.get_limitation_wrist_circle(goal_pose)
        intersection_circle = self.get_intersection_circle(goal_pose)
        if intersection_circle != []:
            self.intersection_circle = intersection_circle
            intervalle = self.are_circles_linked(intersection_circle, limitation_wrist_circle)
            if intervalle != []:
                if SHOW_GRAPH:
                    elbow_position = self.get_coordinate_cercle(intersection_circle, intervalle[0])
                    self.show_point(elbow_position, "r")
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
                    self.show_point(goal_pose[0], "g")
                    self.show_point(self.wrist_position, "r")
                    self.show_point(self.shoulder_position, "b")
                    self.show_point(self.torso_pose, "y")
                    self.show_sphere(self.wrist_position, self.forearm_size, "r")
                    self.show_sphere(self.shoulder_position, self.upper_arm_size, "b")
                    if intersection_circle != []:
                        self.show_circle(
                            intersection_circle[0], intersection_circle[1], intersection_circle[2], [[0, 2 * np.pi]], "g"
                        )
                    self.show_circle(
                        limitation_wrist_circle[0],
                        limitation_wrist_circle[1],
                        limitation_wrist_circle[2],
                        [[0, 2 * np.pi]],
                        "y",
                    )
                    plt.show()
                return [True, intervalle, self.get_joints]

            if SHOW_GRAPH:
                self.show_point(goal_pose[0], "g")
                self.show_point(self.wrist_position, "r")
                self.show_point(self.shoulder_position, "b")
                self.show_point(self.torso_pose, "y")
                self.show_sphere(self.wrist_position, self.forearm_size, "r")
                self.show_sphere(self.shoulder_position, self.upper_arm_size, "b")
                if intersection_circle != []:
                    self.show_circle(
                        intersection_circle[0], intersection_circle[1], intersection_circle[2], [[0, 2 * np.pi]], "g"
                    )
                self.show_circle(
                    limitation_wrist_circle[0], limitation_wrist_circle[1], limitation_wrist_circle[2], [[0, 2 * np.pi]], "y"
                )
                plt.show()
            return [False, [], None]

        if SHOW_GRAPH:
            self.show_point(goal_pose[0], "g")
            self.show_point(self.wrist_position, "r")
            self.show_point(self.shoulder_position, "b")
            self.show_point(self.torso_pose, "y")
            self.show_sphere(self.wrist_position, self.forearm_size, "r")
            self.show_sphere(self.shoulder_position, self.upper_arm_size, "b")
            if intersection_circle != []:
                self.show_circle(intersection_circle[0], intersection_circle[1], intersection_circle[2], [[0, 2 * np.pi]], "g")
            self.show_circle(
                limitation_wrist_circle[0], limitation_wrist_circle[1], limitation_wrist_circle[2], [[0, 2 * np.pi]], "y"
            )
            plt.show()

        return [False, [], None]

    def get_intersection_circle(self, goal_pose):
        wrist_in_shoulder_frame = [
            self.wrist_position[0],
            self.wrist_position[1] - self.shoulder_position[1],
            self.wrist_position[2],
        ]
        d = np.sqrt(wrist_in_shoulder_frame[0] ** 2 + wrist_in_shoulder_frame[1] ** 2 + wrist_in_shoulder_frame[2] ** 2)
        if d > self.upper_arm_size + self.forearm_size:
            # print("goal not reachable")
            return []
        Mrot = R.from_euler(
            "xyz",
            [
                0.0,
                -math.asin(wrist_in_shoulder_frame[2] / d),
                math.atan2(wrist_in_shoulder_frame[1], wrist_in_shoulder_frame[0]),
            ],
        )
        radius = (
            1
            / (2 * d)
            * np.sqrt(4 * d**2 * self.upper_arm_size**2 - (d**2 - self.forearm_size**2 + self.upper_arm_size**2) ** 2)
        )
        center = [(d**2 - self.forearm_size**2 + self.upper_arm_size**2) / (2 * d), 0, 0]
        center = Mrot.apply(center)
        center = [center[0], center[1] - 0.2, center[2]]
        normal_vector = [1, 0, 0]
        normal_vector = Mrot.apply(normal_vector)
        return [center, radius, normal_vector]

    def get_limitation_wrist_circle(self, goal_pose):
        normal_vector = [
            self.wrist_position[0] - goal_pose[0][0],
            self.wrist_position[1] - goal_pose[0][1],
            self.wrist_position[2] - goal_pose[0][2],
        ]
        radius = np.sin(np.radians(self.wrist_limit)) * self.forearm_size
        vector = normal_vector / np.linalg.norm(normal_vector) * np.sqrt(self.forearm_size**2 - radius**2)
        center = self.wrist_position + vector
        return [center, radius, normal_vector]

    def get_wrist_position(self, goal_pose):
        Mrot = R.from_euler("xyz", goal_pose[1])
        wrist_pos = Mrot.apply([0.0, 0.0, self.gripper_size])
        wrist_pos = [wrist_pos[0] + goal_pose[0][0], wrist_pos[1] + goal_pose[0][1], wrist_pos[2] + goal_pose[0][2]]
        return wrist_pos

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        if np.all(np.isclose(vec1, vec2)):
            return np.eye(3)
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
        return rotation_matrix

    def are_circles_linked(self, intersection_circle, limitation_wrist_circle):
        [p1, r1, n1] = limitation_wrist_circle
        [p2, r2, n2] = intersection_circle

        p1 = np.array([p1[0] - self.wrist_position[0], p1[1] - self.wrist_position[1], p1[2] - self.wrist_position[2]])
        p2 = np.array([p2[0] - self.wrist_position[0], p2[1] - self.wrist_position[1], p2[2] - self.wrist_position[2]])
        n2 = np.array(n2)
        n1 = np.array(n1)

        Rmat_intersection = self.rotation_matrix_from_vectors(np.array([1, 0, 0]), n2)
        Tmat_intersection = self.make_homogenous_matrix(p2, Rmat_intersection)
        Tmat_intersection = np.array(
            [
                [Rmat_intersection[0][0], Rmat_intersection[0][1], Rmat_intersection[0][2], p2[0]],
                [Rmat_intersection[1][0], Rmat_intersection[1][1], Rmat_intersection[1][2], p2[1]],
                [Rmat_intersection[2][0], Rmat_intersection[2][1], Rmat_intersection[2][2], p2[2]],
                [0, 0, 0, 1],
            ]
        )
        Rmat_intersection_t = Rmat_intersection.T
        torso_in_intersection_frame = np.dot(-Rmat_intersection_t, p2)
        Tmat_intersection_t = self.make_homogenous_matrix(torso_in_intersection_frame, Rmat_intersection_t)

        Rmat_limitation = self.rotation_matrix_from_vectors(np.array([1, 0, 0]), n1)
        Rmat_limitation_t = Rmat_limitation.T
        torso_in_wrist_limitation_frame = np.dot(-Rmat_limitation_t, p1)
        Tmat_limitation_t = self.make_homogenous_matrix(torso_in_wrist_limitation_frame, Rmat_limitation_t)

        center1 = np.array([p1[0], p1[1], p1[2], 1])
        center2 = np.array([p2[0], p2[1], p2[2], 1])
        center1_in_sphere_frame = np.dot(Tmat_intersection_t, center1)
        n1_in_sphere_frame = np.dot(Rmat_intersection_t, n1)

        if np.any(n1 != 0):
            n1 = n1 / np.linalg.norm(n1)
        if np.any(n2 != 0):
            n2 = n2 / np.linalg.norm(n2)

        if np.all(np.abs(n2 - n1) < 0.0000001) or np.all(np.abs(n2 + n1) < 0.0000001):
            # print("concurrent or parallel")
            if (center1_in_sphere_frame[0] > 0 and n1_in_sphere_frame[0] < 0) or (
                center1_in_sphere_frame[0] < 0 and n1_in_sphere_frame[0] > 0
            ):
                return [0, 2 * np.pi]
            else:
                return []
        else:
            # Find the line of intersection of the planes
            q, v = self.points_of_nearest_approach(p1, n1, p2, n2)
            if q is None:
                if (center1_in_sphere_frame[0] > 0 and n1_in_sphere_frame[0] < 0) or (
                    center1_in_sphere_frame[0] < 0 and n1_in_sphere_frame[0] > 0
                ):
                    # print("no points of nearest approach, goal_pose", self.goal_pose)
                    return [0, 2 * np.pi]
                else:
                    # print("no points of nearest approach, goal_pose", self.goal_pose)
                    return []
            points = self.intersection_circle_line_3d_vd(p1, r1, v, q)
            if points is None:
                if (center1_in_sphere_frame[0] > 0 and n1_in_sphere_frame[0] < 0) or (
                    center1_in_sphere_frame[0] < 0 and n1_in_sphere_frame[0] > 0
                ):
                    return [0, 2 * np.pi]
                else:
                    return []
            if SHOW_GRAPH:
                for point in points:
                    plt.plot(
                        point[0] + self.wrist_position[0],
                        point[1] + self.wrist_position[1],
                        point[2] + self.wrist_position[2],
                        "ro",
                    )

            if len(points) == 1:
                point = [points[0][0], points[0][1], points[0][2], 1]
                point_in_sphere_frame = np.dot(Tmat_intersection_t, point)
                angle = math.atan2(point_in_sphere_frame[2], point_in_sphere_frame[1])
                if angle < 0:
                    angle = angle + 2 * np.pi
                intervalle = [angle, angle]
                return intervalle

            if len(points == 2):
                point1 = [points[0][0], points[0][1], points[0][2], 1]
                point2 = [points[1][0], points[1][1], points[1][2], 1]
                point1_in_sphere_frame = np.dot(Tmat_intersection_t, point1)
                self.intersection = (
                    point1[0] + self.wrist_position[0],
                    point1[1] + self.wrist_position[1],
                    point1[2] + self.wrist_position[2],
                )

                point2_in_sphere_frame = np.dot(Tmat_intersection_t, point2)
                angle1 = math.atan2(point1_in_sphere_frame[2], point1_in_sphere_frame[1])
                angle2 = math.atan2(point2_in_sphere_frame[2], point2_in_sphere_frame[1])

                if angle1 < 0:
                    angle1 = angle1 + 2 * np.pi
                if angle2 < 0:
                    angle2 = angle2 + 2 * np.pi

                [angle1, angle2] = sorted([angle1, angle2])
                angle_test = (angle1 + angle2) / 2
                test_point = np.array([0, math.cos(angle_test) * r2, math.sin(angle_test) * r2, 1])
                test_point = np.dot(Tmat_intersection, test_point)
                # test_point = [test_point[0] + self.wrist_position[0], test_point[1] + self.wrist_position[1], test_point[2] + self.wrist_position[2],1]
                if SHOW_GRAPH:
                    self.ax.plot(
                        test_point[0] + self.wrist_position[0],
                        test_point[1] + self.wrist_position[1],
                        test_point[2] + self.wrist_position[2],
                        "ro",
                    )

                test_point_in_wrist_frame = np.dot(Tmat_limitation_t, test_point)

                if test_point_in_wrist_frame[0] > 0:
                    intervalle = [angle1, angle2]
                else:
                    intervalle = [angle2, np.pi * 2 + angle1]
            return intervalle

    def intersection_point(self, v1, p01, v2, p02):
        A = np.vstack((v1, -v2)).T
        b = p02 - p01
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        print(A)
        print(b)
        print(params)

        if np.all(np.isclose(params, params[0])):
            return None

        intersection = v1 * params[0] + p01
        return intersection

    def points_of_nearest_approach(self, p1, n1, p2, n2):
        v = np.cross(n1, n2)
        v = v / np.linalg.norm(v)
        vect1 = np.cross(v, n1)
        vect2 = np.cross(v, n2)
        q = self.intersection_point(vect1, p1, vect2, p2)
        return q, v

    def intersection_circle_line_3d_vd(self, center, radius, direction, point_on_line):
        a = np.dot(direction, direction)
        b = 2 * np.dot(direction, point_on_line - center)
        c = np.dot(point_on_line - center, point_on_line - center) - radius**2

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return None
        elif discriminant == 0:
            t = -b / (2 * a)
            intersection = point_on_line + t * direction
            return np.array([intersection])
        else:
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)
            intersection1 = point_on_line + t1 * direction
            intersection2 = point_on_line + t2 * direction
        return np.vstack((intersection1, intersection2))

    def get_coordinate_cercle(self, intersection_circle, theta):
        Rmat = self.rotation_matrix_from_vectors(np.array([1, 0, 0]), intersection_circle[2])
        Tmat = np.array(
            [
                [Rmat[0][0], Rmat[0][1], Rmat[0][2], intersection_circle[0][0]],
                [Rmat[1][0], Rmat[1][1], Rmat[1][2], intersection_circle[0][1]],
                [Rmat[2][0], Rmat[2][1], Rmat[2][2], intersection_circle[0][2]],
                [0, 0, 0, 1],
            ]
        )
        x = 0
        y = intersection_circle[1] * np.cos(theta)
        z = intersection_circle[1] * np.sin(theta)
        P = np.array([x, y, z, 1])
        P = np.dot(Tmat, P)
        return P

    def make_transformation_matrix(self, position, orientation):
        Mrot = (R.from_euler("xyz", orientation)).as_matrix()
        T = np.array(
            [
                [Mrot[0][0], Mrot[0][1], Mrot[0][2], position[0]],
                [Mrot[1][0], Mrot[1][1], Mrot[1][2], position[1]],
                [Mrot[2][0], Mrot[2][1], Mrot[2][2], position[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return T

    def make_homogenous_matrix(self, position, rotation_matrix):
        return [
            [rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], position[0]],
            [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], position[1]],
            [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def get_joints(self, theta):
        elbow_position = self.get_coordinate_cercle(self.intersection_circle, theta)
        wrist_position = self.wrist_position
        tip_position = self.goal_pose[0]
        goal_orientation = self.goal_pose[1]
        # print("torso_position", self.torso_position)
        # print("shoulder_position", self.shoulder_position)
        # print("elbow_position", elbow_position)
        # print("wrist_position", wrist_position)
        # print("tip_position", tip_position)
        # print("-------------------")

        shoulder_rotation_matrix = R.from_euler("xyz", np.radians(self.shoulder_orientation_offset))
        offset_rotation_matrix = R.from_euler("xyz", [0.0, np.pi / 2, 0.0])
        shoulder_rotation_matrix = shoulder_rotation_matrix * offset_rotation_matrix
        shoulder_rotation_matrix = shoulder_rotation_matrix.as_matrix()
        shoulder_rotation_matrix_t = shoulder_rotation_matrix.T
        torso_in_shoulder_frame = np.dot(-shoulder_rotation_matrix_t, self.shoulder_position)
        T_torso_shoulder = self.make_homogenous_matrix(torso_in_shoulder_frame, shoulder_rotation_matrix_t)
        elbow_in_shoulder = np.dot(T_torso_shoulder, [elbow_position[0], elbow_position[1], elbow_position[2], 1.0])
        alpha_shoulder = np.arcsin(-elbow_in_shoulder[2] / np.sqrt(elbow_in_shoulder[2] ** 2 + elbow_in_shoulder[0] ** 2))
        if elbow_in_shoulder[0] < 0:
            alpha_shoulder = np.pi - alpha_shoulder

        Ry = R.from_euler("xyz", [0.0, -alpha_shoulder, 0.0]).as_matrix()
        Ty = self.make_homogenous_matrix([0.0, 0.0, 0.0], Ry)
        T_shoulder_elbow = np.dot(Ty, T_torso_shoulder)

        elbow_in_shoulder_bis = np.dot(T_shoulder_elbow, [elbow_position[0], elbow_position[1], elbow_position[2], 1.0])
        x_bis = elbow_in_shoulder_bis[0]
        beta_shoulder = np.arccos(x_bis / self.upper_arm_size)
        if elbow_in_shoulder[1] < 0:
            beta_shoulder = -beta_shoulder

        Rz = R.from_euler("xyz", [0.0, 0.0, -beta_shoulder]).as_matrix()
        Tz = self.make_homogenous_matrix([0.0, 0.0, 0.0], Rz)
        T_shoulder_elbow = np.dot(Tz, T_shoulder_elbow)

        T_torso_elbow = T_shoulder_elbow
        T_torso_elbow[0][3] -= self.upper_arm_size

        wrist_in_elbow = np.dot(T_torso_elbow, [wrist_position[0], wrist_position[1], wrist_position[2], 1.0])
        alpha_elbow = -np.pi / 2 + math.atan2(wrist_in_elbow[2], -wrist_in_elbow[1])

        Rx = R.from_euler("xyz", [alpha_elbow, 0.0, 0.0]).as_matrix()
        Tx = self.make_homogenous_matrix([0.0, 0.0, 0.0], Rx)
        T_shoulder_elbow = np.dot(Tx, T_torso_elbow)

        wrist_in_elbow_bis = np.dot(T_shoulder_elbow, [wrist_position[0], wrist_position[1], wrist_position[2], 1.0])

        beta_elbow = -np.arcsin(wrist_in_elbow_bis[2] / np.sqrt(wrist_in_elbow_bis[0] ** 2 + wrist_in_elbow_bis[2] ** 2))
        if wrist_in_elbow_bis[0] < 0:
            beta_elbow = np.pi - beta_elbow

        Ry = R.from_euler("xyz", [0.0, -beta_elbow, 0.0]).as_matrix()
        Ty = self.make_homogenous_matrix([0.0, 0.0, 0.0], Ry)
        T_shoulder_elbow = np.dot(Ty, T_shoulder_elbow)

        T_torso_wrist = T_shoulder_elbow
        T_torso_wrist[0][3] -= self.forearm_size

        tip_in_wrist = np.dot(T_torso_wrist, [tip_position[0], tip_position[1], tip_position[2], 1.0])

        # Get wrist yaw
        beta_wrist = np.pi - math.atan2(tip_in_wrist[1], -tip_in_wrist[0])

        # --- Get wrist pose in new wrist frame ---
        Ry = R.from_euler("xyz", [0.0, 0.0, -beta_wrist]).as_matrix()
        Ty = self.make_homogenous_matrix([0.0, 0.0, 0.0], Ry)
        T_torso_wrist = np.dot(Ty, T_torso_wrist)

        wrist_in_elbow_bis = np.dot(T_torso_wrist, [tip_position[0], tip_position[1], tip_position[2], 1.0])

        # Get wrist pitch
        alpha_wrist = np.arcsin(wrist_in_elbow_bis[2] / np.sqrt(wrist_in_elbow_bis[0] ** 2 + wrist_in_elbow_bis[2] ** 2))

        Ry = R.from_euler("xyz", [0.0, alpha_wrist, 0.0]).as_matrix()
        Ty = self.make_homogenous_matrix([0.0, 0.0, 0.0], Ry)
        T_torso_wrist = np.dot(Ty, T_torso_wrist)

        T_torso_tip = T_torso_wrist
        T_torso_tip[0][3] -= self.gripper_size

        mrot = R.from_euler("xyz", goal_orientation)
        wrist_pos = mrot.apply([0.1, 0.0, 0.0])
        wrist_pos = [wrist_pos[0] + tip_position[0], wrist_pos[1] + tip_position[1], wrist_pos[2] + tip_position[2]]

        x_in_wrist = np.dot(T_torso_tip, [wrist_pos[0], wrist_pos[1], wrist_pos[2], 1.0])

        gamma_wrist = -math.atan2(x_in_wrist[1], x_in_wrist[2])

        joints = [alpha_shoulder, beta_shoulder, alpha_elbow, beta_elbow, beta_wrist, alpha_wrist, gamma_wrist]

        # norm = np.sqrt(
        #     (self.shoulder_position[0] - self.wrist_position[0]) ** 2
        #     + (self.shoulder_position[1] - self.wrist_position[1]) ** 2
        #     + (self.shoulder_position[2] - self.wrist_position[2]) ** 2
        # )
        # wrist = np.array(self.wrist_position)
        # shoulder = np.array(self.shoulder_position)
        # norm = np.linalg.norm(shoulder - wrist)
        # print(norm)
        # print(joints[3])
        # print(joints[3]/norm)
        # print(2*0.28* np.sin(joints[3]/2))
        return joints

    # ----------------------- show functions -----------------------

    def show_point(self, point, color):
        self.ax.plot(point[0], point[1], point[2], "o", color=color)

    def show_circle(self, center, radius, normal_vector, intervalles, color):
        theta = []
        for intervalle in intervalles:
            angle = np.linspace(intervalle[0], intervalle[1], 100)
            for a in angle:
                theta.append(a)

        y = radius * np.cos(theta)
        z = radius * np.sin(theta)
        x = np.zeros(len(theta))
        Rmat = self.rotation_matrix_from_vectors(np.array([1, 0, 0]), normal_vector)
        Tmat = np.array(
            [
                [Rmat[0][0], Rmat[0][1], Rmat[0][2], center[0]],
                [Rmat[1][0], Rmat[1][1], Rmat[1][2], center[1]],
                [Rmat[2][0], Rmat[2][1], Rmat[2][2], center[2]],
                [0, 0, 0, 1],
            ]
        )

        x2 = np.zeros(len(theta))
        y2 = np.zeros(len(theta))
        z2 = np.zeros(len(theta))
        for k in range(len(theta)):
            p = [x[k], y[k], z[k], 1]
            p2 = np.dot(Tmat, p)
            x2[k] = p2[0]
            y2[k] = p2[1]
            z2[k] = p2[2]
        self.ax.plot(center[0], center[1], center[2], "o", color=color)
        self.ax.plot(x2, y2, z2, color)

    def show_sphere(self, center, radius, color):
        u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
        x = np.cos(u) * np.sin(v) * radius + center[0]
        y = np.sin(u) * np.sin(v) * radius + center[1]
        z = np.cos(v) * radius + center[2]
        self.ax.plot_wireframe(x, y, z, color=color, alpha=0.2)


if __name__ == "__main__":
    ik = IK_symbolic()

    goal_position = [0.3, -0.1, 0.1]
    goal_orientation = [20, -50, 20]
    goal_orientation = np.deg2rad(goal_orientation)
    goal_pose = [goal_position, goal_orientation]

    # T = ik.make_transformation_matrix(goal_position, goal_orientation)

    # start_time = time.time()
    # for i in range(1000000):
    #     Tt = np.linalg.inv(T)
    # end_time = time.time()
    # print("np.linalg.inv", end_time - start_time)

    # start_time = time.time()
    # for i in range(1000000):
    #     Rt = T[0:3, 0:3].T
    #     P = np.dot(-Rt, T[0:3, 3])
    #     Tt = ik.make_homogenous_matrix(P, Rt)
    # end_time = time.time()
    # print("make_homogenous_matrix", end_time - start_time)

    result = ik.is_reachable(goal_pose)
    # if result[0]:
    #     joints = result[2](result[1][0])
    #     print(joints)
    #     joints2 = ik.get_joints2(result[1][0])
    #     print(joints2)
    #     for i in range(len(joints)):
    #         print(joints[i]-joints2[i])

    # start_time = time.time()
    # for i in range(10000):
    #     ik.get_joints2(result[1][0])
    # end_time = time.time()
    # print("get_joints2", end_time - start_time)

    # start_time = time.time()
    # for i in range(10000):
    #     ik.get_joints(result[1][0])
    # end_time = time.time()
    # print("get_joints", end_time - start_time)

    # start_time = time.time()
    # for i in range(10000):
    #     ik.is_reachable(goal_pose)
    # end_time = time.time()
    # print("is_reachable", end_time - start_time)

    # start_time = time.time()
    # for i in range(10000):
    #     ik.get_wrist_position(goal_pose)
    # end_time = time.time()
    # print("get_wrist_position", end_time - start_time)

    # ik.wrist_position = ik.get_wrist_position(goal_pose)

    # start_time = time.time()
    # for i in range(10000):
    #     ik.get_limitation_wrist_circle(goal_pose)
    # end_time = time.time()
    # print("get_limitation_wrist_circle", end_time - start_time)

    # start_time = time.time()
    # for i in range(10000):
    #     ik.get_intersection_circle(goal_pose)
    # end_time = time.time()
    # print("get_intersection_circle", end_time - start_time)

    # intersection_circle = ik.get_intersection_circle(goal_pose)
    # limitation_wrist_circle = ik.get_limitation_wrist_circle(goal_pose)

    # start_time = time.time()
    # for i in range(10000):
    #     ik.are_circles_linked(intersection_circle, limitation_wrist_circle)
    # end_time = time.time()
    # print("are_circles_linked", end_time - start_time)

    # limitation_wrist_circle = ik.get_limitation_wrist_circle(goal_pose)
    # intersection_circle = ik.get_intersection_circle(goal_pose)

    # start_time = time.time()
    # for i in range(10000):
    #     ik.rotation_matrix_from_vectors(np.array([1, 0, 0]), limitation_wrist_circle[2])
    # end_time = time.time()
    # print("rotation_matrix_from_vectors", end_time - start_time)
    # [p1, r1, n1] = limitation_wrist_circle
    # [p2, r2, n2] = intersection_circle

    # Rmat_limitation = ik.rotation_matrix_from_vectors(np.array([1, 0, 0]), n1)
    # Tmat_limitation = np.array(
    #     [
    #         [Rmat_limitation[0][0], Rmat_limitation[0][1], Rmat_limitation[0][2], p1[0]],
    #         [Rmat_limitation[1][0], Rmat_limitation[1][1], Rmat_limitation[1][2], p1[1]],
    #         [Rmat_limitation[2][0], Rmat_limitation[2][1], Rmat_limitation[2][2], p1[2]],
    #         [0, 0, 0, 1],
    #     ]
    # )

    # start_time = time.time()
    # for i in range(10000):
    #     Tmat_limitation = np.array(
    #         [0,0,0,1]
    #     )
    # end_time = time.time()
    # print("np.array", end_time - start_time)

    # P = [0.0, 0.0, 0.0, 1.0]
    # start_time = time.time()
    # for i in range(10000):
    #     np.dot(Tmat_limitation, P)
    # end_time = time.time()
    # print("np.dot", end_time - start_time)

    # start_time = time.time()
    # for i in range(10000):
    #     ik.points_of_nearest_approach(limitation_wrist_circle[0], limitation_wrist_circle[2], intersection_circle[0], intersection_circle[2])
    # end_time = time.time()
    # print("points_of_nearest_approach", end_time - start_time)

    # q,v = ik.points_of_nearest_approach(limitation_wrist_circle[0], limitation_wrist_circle[2], intersection_circle[0], intersection_circle[2])

    # start_time = time.time()
    # for i in range(10000):
    #     ik.intersection_circle_line_3d_vd(limitation_wrist_circle[0], limitation_wrist_circle[1], v, q)
    # end_time = time.time()
    # print("intersection_circle_line_3d_vd", end_time - start_time)
