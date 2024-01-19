from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
import copy
import tf_transformations
import math
import numpy as np
import matplotlib.pyplot as plt

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
    ):
        self.grasp_marker_tip_len = grasp_marker_tip_len
        self.grasp_marker_width = grasp_marker_width
        self.upper_arm_size = upper_arm_size
        self.forearm_size = forearm_size
        self.gripper_size = gripper_size
        self.x_offset = x_offset
        self.wrist_limit = wrist_limit
        self.torso_pose = [0.0, 0.0, 0.0]
        self.shoulder_posistion = [0.0, -0.2, 0.0]
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
            # print(intervalle)
            if intervalle != []:
                # print(elbow_position)
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
                        [elbow_position[0], self.shoulder_posistion[0]],
                        [elbow_position[1], self.shoulder_posistion[1]],
                        [elbow_position[2], self.shoulder_posistion[2]],
                        "r",
                    )
                    self.show_point(goal_pose[0], "g")
                    # self.ax.plot([goal_pose[0][0], goal_pose[0][0]+goal_pose[1][0]/10],[goal_pose[0][1], goal_pose[0][1]+goal_pose[1][1]/10],[goal_pose[0][2], goal_pose[0][2]+goal_pose[1][2]/10],  'g')
                    self.show_point(self.wrist_position, "r")
                    self.show_point(self.shoulder_posistion, "b")
                    self.show_point(self.torso_pose, "y")
                    self.show_sphere(self.wrist_position, self.forearm_size, "r")
                    self.show_sphere(self.shoulder_posistion, self.upper_arm_size, "b")
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
                # self.get_joints(limits[0])
                return [True, intervalle, self.get_joints]
            if SHOW_GRAPH:
                self.show_point(goal_pose[0], "g")
                # self.ax.plot([goal_pose[0][0], goal_pose[0][0]+goal_pose[1][0]/10],[goal_pose[0][1], goal_pose[0][1]+goal_pose[1][1]/10],[goal_pose[0][2], goal_pose[0][2]+goal_pose[1][2]/10],  'g')
                self.show_point(self.wrist_position, "r")
                self.show_point(self.shoulder_posistion, "b")
                self.show_point(self.torso_pose, "y")
                self.show_sphere(self.wrist_position, self.forearm_size, "r")
                self.show_sphere(self.shoulder_posistion, self.upper_arm_size, "b")
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
            # self.ax.plot([goal_pose[0][0], goal_pose[0][0]+goal_pose[1][0]/10],[goal_pose[0][1], goal_pose[0][1]+goal_pose[1][1]/10],[goal_pose[0][2], goal_pose[0][2]+goal_pose[1][2]/10],  'g')
            self.show_point(self.wrist_position, "r")
            self.show_point(self.shoulder_posistion, "b")
            self.show_point(self.torso_pose, "y")
            self.show_sphere(self.wrist_position, self.forearm_size, "r")
            self.show_sphere(self.shoulder_posistion, self.upper_arm_size, "b")
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
            self.wrist_position[1] - self.shoulder_posistion[1],
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
        # print(center, radius, normal_vector)
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

    # def get_wrist_position(self, goal_pose):
    #     Mrot = self.rotation_matrix_from_vectors([1,0,0], goal_pose[1])
    #     P = [-self.gripper_size,0.,0.]
    #     P = np.array(P)
    #     P = np.dot(Mrot, P)
    #     wrist_position = [P[0] + goal_pose[0][0], P[1] + goal_pose[0][1], P[2] + goal_pose[0][2]]

    #     return wrist_position

    def get_wrist_position(self, goal_pose):
        Mrot = R.from_euler("xyz", goal_pose[1])
        # print("Mrot", Mrot.as_matrix())
        wrist_pos = Mrot.apply([0.0, 0.0, self.gripper_size])
        wrist_pos = [wrist_pos[0] + goal_pose[0][0], wrist_pos[1] + goal_pose[0][1], wrist_pos[2] + goal_pose[0][2]]
        return wrist_pos

    def show_point(self, point, color):
        self.ax.plot(point[0], point[1], point[2], "o", color=color)

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

    def show_circle(self, center, radius, normal_vector, intervalles, color):
        theta = []
        for intervalle in intervalles:
            # print(intervalle)
            angle = np.linspace(intervalle[0], intervalle[1], 100)
            # print("test")
            for a in angle:
                theta.append(a)

        y = radius * np.cos(theta)
        z = radius * np.sin(theta)
        x = np.zeros(len(theta))
        Rmat = self.rotation_matrix_from_vectors(np.array([1, 0, 0]), normal_vector)
        vect = np.dot(Rmat, np.array([0, 0, 1]))
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
        # self.ax.plot(0, 0, 0, 'o', color=color)

    def are_circles_linked(self, intersection_circle, limitation_wrist_circle):
        [p1, r1, n1] = limitation_wrist_circle
        [p2, r2, n2] = intersection_circle

        p1 = np.array([p1[0] - self.wrist_position[0], p1[1] - self.wrist_position[1], p1[2] - self.wrist_position[2]])
        p2 = np.array([p2[0] - self.wrist_position[0], p2[1] - self.wrist_position[1], p2[2] - self.wrist_position[2]])
        n2 = np.array(n2)
        n1 = np.array(n1)

        Rmat_intersection = self.rotation_matrix_from_vectors(np.array([1, 0, 0]), n2)
        Tmat_intersection = np.array(
            [
                [Rmat_intersection[0][0], Rmat_intersection[0][1], Rmat_intersection[0][2], p2[0]],
                [Rmat_intersection[1][0], Rmat_intersection[1][1], Rmat_intersection[1][2], p2[1]],
                [Rmat_intersection[2][0], Rmat_intersection[2][1], Rmat_intersection[2][2], p2[2]],
                [0, 0, 0, 1],
            ]
        )

        Rmat_limitation = self.rotation_matrix_from_vectors(np.array([1, 0, 0]), n1)
        Tmat_limitation = np.array(
            [
                [Rmat_limitation[0][0], Rmat_limitation[0][1], Rmat_limitation[0][2], p1[0]],
                [Rmat_limitation[1][0], Rmat_limitation[1][1], Rmat_limitation[1][2], p1[1]],
                [Rmat_limitation[2][0], Rmat_limitation[2][1], Rmat_limitation[2][2], p1[2]],
                [0, 0, 0, 1],
            ]
        )

        Tmat_intersection_t = np.linalg.inv(Tmat_intersection)
        Tmat_limitation_t = np.linalg.inv(Tmat_limitation)

        n2_bis = np.dot(Tmat_intersection_t, np.array([n2[0], n2[1], n2[2], 0]))
        # print("n2_bis", n2_bis)

        center1 = np.array([p1[0], p1[1], p1[2], 1])
        center2 = np.array([p2[0], p2[1], p2[2], 1])

        # center1ee = np.dot(Tmat_limitation_t, center1)
        # center2ee = np.dot(Tmat_intersection_t, center2)
        # print("center1ee", center1ee)
        # print("center2ee", center2ee)

        center1_in_sphere_frame = np.dot(Tmat_intersection_t, center1)
        center2_in_sphere_frame = np.dot(Tmat_intersection_t, center2)
        # print("center1 : ",center1_in_sphere_frame)
        # print("center2 : ", center2_in_sphere_frame)
        n1_in_sphere_frame = np.dot(Tmat_intersection_t, np.array([n1[0], n1[1], n1[2], 0]))

        if np.any(n1 != 0):
            n1 = n1 / np.linalg.norm(n1)
        if np.any(n2 != 0):
            n2 = n2 / np.linalg.norm(n2)
        #     print(np.linalg.norm(n2))
        # print(n1, n2)

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

                # print(test_point)

                # Tmat_limitation_t = np.linalg.inv(Tmat_limitation)
                test_point_in_wrist_frame = np.dot(Tmat_limitation_t, test_point)

                if test_point_in_wrist_frame[0] > 0:
                    intervalle = [angle1, angle2]
                else:
                    intervalle = [angle2, np.pi * 2 + angle1]
                # print(intervalle)
                # print("------------")
            return intervalle

    def intersection_point(self, v1, p01, v2, p02):
        A = np.vstack((v1, -v2)).T
        b = p02 - p01
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Vérifier si les vecteurs sont colinéaires (pas d'intersection unique)
        if np.all(np.isclose(params, params[0])):
            return None

        intersection = v1 * params[0] + p01
        return intersection

    def points_of_nearest_approach(self, p1, n1, p2, n2):
        v = np.cross(n1, n2)
        A = np.array([n1, -n2, v]).T
        b = p2 - p1
        s, t, _ = np.linalg.lstsq(A, b, rcond=None)[0]
        q1 = p1 + s * (np.cross(v, n1))
        q2 = p2 + t * (np.cross(v, n2))
        vect1 = np.cross(v, n1)
        v = v / np.linalg.norm(v)
        # ax.plot([center1[0], center1[0]+vect1[0]], [center1[1], center1[1]+vect1[1]], [center1[2], center1[2]+vect1[2]], 'r' )
        vect2 = np.cross(v, n2)
        # ax.plot([center2[0], center2[0]+vect2[0]], [center2[1], center2[1]+vect2[1]], [center2[2], center2[2]+vect2[2]], 'r' )
        q = self.intersection_point(vect1, p1, vect2, p2)
        # print(vect1, vect2)
        # if q is not None:
        #     ax.plot(q[0], q[1], q[2], 'ro')
        #     ax.plot([q[0]-2*v[0], q[0]+v[0]*2], [q[1]-2*v[1], q[1]+v[1]*2], [q[2]-2*v[2], q[2]+2*v[2]], 'r' )
        return q, v

    def intersection_circle_line_3d_vd(self, center, radius, direction, point_on_line):
        # Équation du cercle : (x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2 = r^2
        # Équation de la droite : p(t) = point_on_line + t * direction

        a = np.dot(direction, direction)  # Coefficient devant t^2
        b = 2 * np.dot(direction, point_on_line - center)  # Coefficient devant t
        c = np.dot(point_on_line - center, point_on_line - center) - radius**2  # Terme constant

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            # Pas d'intersection
            return None
        elif discriminant == 0:
            # Une seule intersection
            t = -b / (2 * a)
            intersection = point_on_line + t * direction
            return np.array([intersection])
        else:
            # Deux intersections
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
        # print("position", P)

        return P

    def make_transformation_matrix(self, position, orientation):
        Mrot = (R.from_euler("xyz", orientation)).as_matrix()
        # print("Mrot ", Mrot[0][0])
        # print(position)
        T = np.array(
            [
                [Mrot[0][0], Mrot[0][1], Mrot[0][2], position[0]],
                [Mrot[1][0], Mrot[1][1], Mrot[1][2], position[1]],
                [Mrot[2][0], Mrot[2][1], Mrot[2][2], position[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return T

    def get_joints(self, theta):
        torso_position = self.torso_pose
        shoulder_position = self.shoulder_posistion
        elbow_position = self.get_coordinate_cercle(self.intersection_circle, theta)
        wrist_position = self.wrist_position
        tip_position = self.goal_pose[0]
        goal_orientation = self.goal_pose[1]
        # print("torso_position", torso_position)
        # print("shoulder_position", shoulder_position)
        # print("elbow_position", elbow_position)
        # print("wrist_position", wrist_position)
        # print("tip_position", tip_position)
        # print("-------------------")

        T_shoulder_torso = self.make_transformation_matrix(shoulder_position, [0.0, 0.0, 0.0])

        # --- Get elbow pose in shoulder frame ---
        elbow_in_shoulder = np.dot(
            np.linalg.inv(T_shoulder_torso), [elbow_position[0], elbow_position[1], elbow_position[2], 1.0]
        )
        d_elbow_in_shoulder = np.sqrt(
            elbow_in_shoulder[0] * elbow_in_shoulder[0]
            + elbow_in_shoulder[1] * elbow_in_shoulder[1]
            + elbow_in_shoulder[2] * elbow_in_shoulder[2]
        )

        # Get shoulder pitch
        alpha_shoulder = -np.arcsin(
            elbow_in_shoulder[0]
            / np.sqrt(elbow_in_shoulder[0] * elbow_in_shoulder[0] + elbow_in_shoulder[2] * elbow_in_shoulder[2])
        )
        if elbow_in_shoulder[2] > 0:
            alpha_shoulder = np.pi - alpha_shoulder

        rotation_y = [0.0, alpha_shoulder, 0.0]
        Ty = self.make_transformation_matrix([0.0, 0.0, 0.0], rotation_y)
        T_shoulder_elbow = np.dot(T_shoulder_torso, Ty)

        # --- Get elbow pose in new shoulder frame ---
        elbow_in_shoulder_bis = np.dot(
            np.linalg.inv(T_shoulder_elbow), [elbow_position[0], elbow_position[1], elbow_position[2], 1.0]
        )
        x_bis = -elbow_in_shoulder_bis[2]

        # Get shoulder roll
        beta_shoulder = np.arccos(x_bis / d_elbow_in_shoulder)
        if elbow_in_shoulder[1] < 0:
            beta_shoulder = -beta_shoulder

        # print("alpha_shoulder", alpha_shoulder)
        # print("beta_shoulder", beta_shoulder)

        roll_axe = R.from_euler("xyz", [0.0, 90.0, 0.0], degrees=True)
        rot_y = R.from_euler("xyz", [0.0, alpha_shoulder, 0.0])
        rot_x = R.from_euler("xyz", [0.0, 0.0, beta_shoulder])
        orientation = roll_axe * rot_y
        orientation = orientation * rot_x

        # --------------------------------------------------------

        orientation1 = orientation.as_matrix()
        T_torso_elbow = np.array(
            [
                [orientation1[0][0], orientation1[0][1], orientation1[0][2], elbow_position[0]],
                [orientation1[1][0], orientation1[1][1], orientation1[1][2], elbow_position[1]],
                [orientation1[2][0], orientation1[2][1], orientation1[2][2], elbow_position[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # --- Get wrist pose in elbow frame ---
        wrist_in_elbow = np.dot(np.linalg.inv(T_torso_elbow), [wrist_position[0], wrist_position[1], wrist_position[2], 1.0])
        # self.markers.markers.append(self.create_sphere(self.apply_offset(wrist_in_elbow[0:3]), ColorRGBA(r=0.9, g=0.3, b=0.3, a=1.), 0.03))

        # Get elbow roll
        alpha_elbow = -np.pi / 2 + math.atan2(wrist_in_elbow[2], -wrist_in_elbow[1])
        rotation_y = R.from_euler("xyz", [-alpha_elbow, 0.0, 0.0])
        Tx = self.make_transformation_matrix([0.0, 0.0, 0.0], [-alpha_elbow, 0.0, 0.0])
        T_shoulder_elbow = np.dot(T_torso_elbow, Tx)

        # --- Get wrist pose in new elbow frame ---
        wrist_in_elbow_bis = np.dot(
            np.linalg.inv(T_shoulder_elbow), [wrist_position[0], wrist_position[1], wrist_position[2], 1.0]
        )

        # Get elbow pitch
        beta_elbow = -np.arcsin(wrist_in_elbow_bis[2] / np.sqrt(wrist_in_elbow_bis[0] ** 2 + wrist_in_elbow_bis[2] ** 2))
        if wrist_in_elbow_bis[0] < 0:
            beta_elbow = np.pi - beta_elbow

        # print("alpha_elbow ", alpha_elbow)
        # print("beta_elbow ", beta_elbow)

        roll_axe = R.from_euler("xyz", [0.0, 90.0, 0.0], degrees=True)
        rot_y = R.from_euler("xyz", [-alpha_elbow, 0.0, 0.0])
        rot_x = R.from_euler("xyz", [0.0, beta_elbow, 0.0])
        orientation = orientation * rot_y
        orientation = orientation * rot_x

        # #--------------------------------------------------------
        # roll_axe = R.from_euler('xyz', [0.0, 90.0, 0.0], degrees=True)
        # orientation = orientation * roll_axe
        orientation1 = orientation.as_matrix()
        T_torso_wrist = np.array(
            [
                [orientation1[0][0], orientation1[0][1], orientation1[0][2], wrist_position[0]],
                [orientation1[1][0], orientation1[1][1], orientation1[1][2], wrist_position[1]],
                [orientation1[2][0], orientation1[2][1], orientation1[2][2], wrist_position[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # --- Get tip pose in wrist frame ---
        tip_in_wrist = np.dot(np.linalg.inv(T_torso_wrist), [tip_position[0], tip_position[1], tip_position[2], 1.0])

        # Get wrist yaw
        beta_wrist = np.pi - math.atan2(tip_in_wrist[1], -tip_in_wrist[0])

        # --- Get wrist pose in new wrist frame ---
        Ty = self.make_transformation_matrix([0.0, 0.0, 0.0], [0.0, 0.0, beta_wrist])
        T_torso_wrist = np.dot(T_torso_wrist, Ty)
        rot_y = R.from_euler("xyz", [0.0, 0.0, beta_wrist])
        wrist_in_elbow_bis = np.dot(np.linalg.inv(T_torso_wrist), [tip_position[0], tip_position[1], tip_position[2], 1.0])

        # Get wrist pitch
        # TODO tester cas avec changements de signes
        alpha_wrist = np.arcsin(wrist_in_elbow_bis[2] / np.sqrt(wrist_in_elbow_bis[0] ** 2 + wrist_in_elbow_bis[2] ** 2))

        Ty = self.make_transformation_matrix([0.0, 0.0, 0.0], [0.0, -alpha_wrist, 0.0])
        T_torso_wrist = np.dot(T_torso_wrist, Ty)

        T_torso_tip = np.copy(T_torso_wrist)
        T_torso_tip[0][3] = tip_position[0]
        T_torso_tip[1][3] = tip_position[1]
        T_torso_tip[2][3] = tip_position[2]

        # TODO : correct gamma_wrist

        mrot = R.from_euler("xyz", goal_orientation)
        wrist_pos = mrot.apply([0.1, 0.0, 0.0])
        wrist_pos = [wrist_pos[0] + tip_position[0], wrist_pos[1] + tip_position[1], wrist_pos[2] + tip_position[2]]
        # self.markers.markers.append(self.create_sphere(self.apply(self.apply_offset(wrist_pos),tip_position), ColorRGBA(r=1., g=0.3, b=0.3, a=1.), 0.03))

        # test= np.dot((T_torso_tip), [0.,0.,0.1,1])
        # self.markers.markers.append(self.create_sphere(self.apply_offset(test[0:3]), ColorRGBA(r=0.9, g=0.3, b=0.3, a=1.), 0.03))

        x_in_wrist = np.dot(np.linalg.inv(T_torso_tip), [wrist_pos[0], wrist_pos[1], wrist_pos[2], 1.0])
        # self.markers.markers.append(self.create_sphere(self.apply_offset(x_in_wrist[0:3]), ColorRGBA(r=1., g=0.3, b=0.3, a=1.), 0.03))

        gamma_wrist = -math.atan2(x_in_wrist[1], x_in_wrist[2])

        rot_x = R.from_euler("xyz", [0.0, -alpha_wrist, 0.0])
        rot_y = R.from_euler("xyz", [0.0, 0.0, beta_wrist])
        orientation = orientation * rot_y
        orientation = orientation * rot_x

        # TODO : correct gamma_wrist

        # gamma_wrist = -orientation.as_euler('xyz')[0] + self.goal_pose[1][0]
        rot_z = R.from_euler("xyz", [gamma_wrist, 0.0, 0.0])
        orientation = orientation * rot_z

        joints = [alpha_shoulder, beta_shoulder, alpha_elbow, beta_elbow, alpha_wrist, beta_wrist, gamma_wrist]
        # print(joints)
        return joints
