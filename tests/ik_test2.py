import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import Header
from grasping_utils.utils import get_grasp_pose_msg_from_quaternion, get_grasp_marker
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
import copy
import tf_transformations
import math
from std_msgs.msg import ColorRGBA
from rclpy.qos import ReliabilityPolicy, HistoryPolicy
import numpy as np
import time
from reachy_placo.ik_reachy_placo import IKReachyQP

GRASP_MARKER_TIP_LEN = 0.2
GRASP_MARKER_WIDTH = 20
UPPER_ARM_SIZE = 0.28
FOREARM_SIZE = 0.28
GRIPPER_SIZE = 0.15  # From the center of the wrist to the tip
X_OFFSET = 3.0


class ArrowPublisher(Node):
    def __init__(self):
        super().__init__("arrow_publisher")
        self.publisher_ = self.create_publisher(PoseStamped, "arrow_topic", 10)
        qos_policy = rclpy.qos.QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_ALL, depth=100)
        self.marker_pub = self.create_publisher(MarkerArray, "reachability_markers", qos_profile=qos_policy)
        self.markers = MarkerArray()
        self.create_timer(0.1, self.main_tick)
        self.id = 1

        # self.ik_reachy = IKReachyQP(
        #     viewer_on=True,
        #     collision_avoidance=False,
        #     parts=["r_arm"],
        #     position_weight=1.9,
        #     orientation_weight=1e-2,
        #     robot_version="reachy_2",
        #     velocity_limit=50.0,
        # )
        # # -> set velocity_limit=30 when using the reachability/IK calls instead of the continuous QP control.
        # # self.ik_reachy.setup(urdf_path="../reachy/new_reachy2_handmade.urdf")
        # self.ik_reachy.setup(urdf_path="../reachy_placo/reachy_placo/reachy/new_new_reachy2_1.urdf")
        # self.ik_reachy.create_tasks()

    def main_tick(self):
        if self.id == 1:
            # pose of tip in torso frame
            # TODO : Gerer le cas ou la pose n'est pas atteignable
            tip_x = 0.3
            tip_y = -0.2
            tip_z = -0.3

            goal_position = [0.3, -0.2, 0.0]
            goal_orientation = [-45, -30, 20]
            goal_orientation = [
                math.radians(goal_orientation[0]),
                math.radians(goal_orientation[1]),
                math.radians(goal_orientation[2]),
            ]
            # goal_position = [0.2,-0.0,0.1]
            # goal_orientation = [2.,2.,-3.]
            # goal_orientation = goal_orientation/np.linalg.norm(goal_orientation)

            goal_pose = [goal_position, goal_orientation]
            print(goal_position)
            # print(tip_x)

            self.show_goal_position(goal_pose)
            self.show_robot_arm(goal_pose)

            colors = [
                ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
            ]
            self.show_frame([0.0, -0.2, 0.0], [0.0, 0.0, 0.0], colors)

            [x, y, z] = self.get_wrist_position(goal_pose)
            [x, y, z] = [x, y + 0.2, z]
            d = np.sqrt(x * x + y * y + z * z)
            Mrot = R.from_euler("xyz", [0.0, -math.asin(z / d), math.atan2(y, x)])

            colors = [
                ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),
                ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),
            ]
            roll_axe = R.from_euler("xyz", [0.0, 0.0, 0.0])
            roll_axe = roll_axe * Mrot
            roll_axe = roll_axe.as_euler("xyz")
            self.show_frame([0.0, -0.2, 0.0], roll_axe, colors)

            # Angle on the solutions circle
            # TODO : Checker si tous les theta sont corrects
            thetha = 0
            elbox_position = self.get_coordinate_cercle(d, thetha)
            elbox_position = Mrot.apply(elbox_position)
            elbox_position = [elbox_position[0], elbox_position[1] - 0.2, elbox_position[2]]
            print("elbox_position ", elbox_position)
            self.markers.markers.append(
                self.create_sphere(self.apply_offset(elbox_position), ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), 0.03)
            )

            # ______________________________________________________

            # get joints position from tip position

            torso_position = [0.0, 0.0, 0.0]
            shoulder_position = [0.0, -0.2, 0.0]
            elbow_position = elbox_position
            wrist_position = self.get_wrist_position(goal_pose)
            tip_position = goal_pose[0]

            # pose to test the transformation matrix
            orientation_test = [0.0, 0.0, 0.0]
            position_test = [0.0, 0.0, -UPPER_ARM_SIZE]
            T_test = self.make_transformation_matrix(position_test, orientation_test)
            orientation_test = [0.0, 0.0, 0.0]
            position_test = [0.0, 0.0, 0.0]
            T_test2 = self.make_transformation_matrix(position_test, orientation_test)
            colors_test = [
                ColorRGBA(r=0.3, g=0.3, b=0.3, a=1.0),
                ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0),
                ColorRGBA(r=0.8, g=0.8, b=0.8, a=1.0),
            ]
            self.show_frame(position_test, orientation_test, colors_test)

            T_shoulder_torso = self.make_transformation_matrix(shoulder_position, [0.0, 0.0, 0.0])
            test_pos, test_rot = self.get_position_orientation_from_transformation(np.dot(T_shoulder_torso, T_test))
            self.show_frame(test_pos, test_rot, colors_test)

            # T_elbow_zero = self.make_transformation_matrix([shoulder_position[0], shoulder_position[1], shoulder_position[2]-UPPER_ARM_SIZE], [0.0, 0.0, 0.0])
            # test_pos, test_rot = self.get_position_orientation_from_transformation(np.dot(T_shoulder_torso, T_test))
            # self.show_frame(test_pos, test_rot, colors_test)

            # --- Get elbow pose in shoulder frame ---
            elbow_in_shoulder = np.dot(
                np.linalg.inv(T_shoulder_torso), [elbow_position[0], elbow_position[1], elbow_position[2], 1.0]
            )
            d_elbow_in_shoulder = np.sqrt(
                elbow_in_shoulder[0] * elbow_in_shoulder[0]
                + elbow_in_shoulder[1] * elbow_in_shoulder[1]
                + elbow_in_shoulder[2] * elbow_in_shoulder[2]
            )
            # self.show_frame(elbow_in_shoulder[0:3], [0.0, 0.0, 0.0], colors_test)

            # print([0.,math.atan2(elbow_in_shoulder[2],elbow_in_shoulder[0]), 0.])
            # print(np.arcsin(elbow_in_shoulder[0]/np.sqrt(elbow_in_shoulder[0]*elbow_in_shoulder[0] + elbow_in_shoulder[2]*elbow_in_shoulder[2])))
            # print(np.arccos(elbow_in_shoulder[2]/np.sqrt(elbow_in_shoulder[0]*elbow_in_shoulder[0] + elbow_in_shoulder[2]*elbow_in_shoulder[2])))
            # print(np.arcsin(elbow_in_shoulder[1]/np.sqrt(elbow_in_shoulder[1]*elbow_in_shoulder[1] + elbow_in_shoulder[0]*elbow_in_shoulder[0])))
            # print(np.arccos(elbow_in_shoulder[0]/np.sqrt(elbow_in_shoulder[1]*elbow_in_shoulder[1] + elbow_in_shoulder[0]*elbow_in_shoulder[0])))
            # print(np.arctan2(elbow_in_shoulder[0],elbow_in_shoulder[1]))
            #  [0.0, 0.0, math.atan2(y, x)])
            # rot2 = R.from_euler('xyz', [0.0, -math.asin(z/d), math.atan2(y, x)])
            # T_se = self.make_transformation_matrix(elbow_in_shoulder, [0.0, -math.asin(elbow_in_shoulder[2]/d_elbow_in_shoulder), math.atan2(elbow_in_shoulder[1], elbow_in_shoulder[0])])
            # test_pos, test_rot = self.get_position_orientation_from_transformation(np.dot(T_shoulder_torso,np.dot(T_se, T_test)))
            # self.show_frame(test_pos, test_rot, colors_test)
            # alpha_shoulder = np.pi + np.arccos(elbow_in_shoulder[2]/np.sqrt(elbow_in_shoulder[0]*elbow_in_shoulder[0] + elbow_in_shoulder[2]*elbow_in_shoulder[2]))

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

            # test_pos, test_rot = self.get_position_orientation_from_transformation(np.dot(T_shoulder_elbow, T_test))
            # test_pos, test_rot = self.get_position_orientation_from_transformation(np.dot(T_shoulder_elbow, T_test2))
            # self.show_frame(test_pos, test_rot, colors_test)

            # --- Get elbow pose in new shoulder frame ---
            elbow_in_shoulder_bis = np.dot(
                np.linalg.inv(T_shoulder_elbow), [elbow_position[0], elbow_position[1], elbow_position[2], 1.0]
            )
            x_bis = -elbow_in_shoulder_bis[2]
            # print("x_bis",x_bis)
            # print(np.arccos(x_bis/d_elbow_in_shoulder))
            # print(np.pi - np.arctan2(np.sqrt(UPPER_ARM_SIZE**2 - x_bis**2), -x_bis))

            # Get shoulder roll
            beta_shoulder = np.arccos(x_bis / d_elbow_in_shoulder)
            if elbow_in_shoulder[1] < 0:
                beta_shoulder = -beta_shoulder

            print("alpha_shoulder", alpha_shoulder)
            print("beta_shoulder", beta_shoulder)

            # rotation_x = [beta_shoulder, 0., 0.]
            # Tx = self.make_transformation_matrix([0.,0.,0.], rotation_x)
            # T_shoulder_elbow = np.dot(T_shoulder_torso,np.dot(T_shoulder_elbow, Tx))
            # test_pos, test_rot = self.get_position_orientation_from_transformation(np.dot(T_shoulder_torso,np.dot(T_shoulder_elbow, T_test)))
            # self.show_frame(test_pos, test_rot, colors_test)

            # ---- Code magique pour l'épaule, modifier les angles alpha et bate de l'epaule permet d'obtenir les mêmes coordonnées du coude que dans meshcat
            # respresenté par une fleche rouge et un point jaune  ------

            # alpha_shoulder = -0.88
            # beta_shoulder = 0.
            roll_axe = R.from_euler("xyz", [0.0, 90.0, 0.0], degrees=True)
            rot_y = R.from_euler("xyz", [0.0, alpha_shoulder, 0.0])
            rot_x = R.from_euler("xyz", [0.0, 0.0, beta_shoulder])
            orientation = roll_axe * rot_y
            orientation = orientation * rot_x
            colors = [
                ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),
                ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),
            ]
            self.show_frame(elbow_position, orientation.as_euler("xyz"), colors)
            shoulder_pos = orientation.apply([0.28, 0.0, 0.0])
            self.markers.markers.append(
                self.create_sphere(
                    self.apply_shoulder(self.apply_offset(shoulder_pos)), ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0), 0.03
                )
            )

            shoulder_orientation = orientation.as_euler("xyz")
            self.markers.markers.append(
                self.create_arrow(
                    self.apply_offset([0.0, -0.2, 0.0]), shoulder_orientation, ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), 0.28, []
                )
            )

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

            # position = [0.,.0, 0.1, 1.]
            # position_elbow = np.dot(T_torso_elbow, position)
            # print(position_elbow)
            # self.markers.markers.append(self.create_sphere(self.apply_offset(position_elbow[0:3]), ColorRGBA(r=0.0, g=0., b=0.0, a=1.), 0.03))

            # orientation_torso_elbow = R.from_euler('xyz', [0.0, -math.asin(z/d), math.atan2(y, x)])
            # colors = [ColorRGBA(r=1.0, g=0., b=1.0, a=1.), ColorRGBA(r=1.0, g=1., b=0.0, a=1.), ColorRGBA(r=.0, g=1., b=1.0, a=1.)]
            # self.show_frame(elbow_position, orientation_torso_elbow.as_euler('xyz'), colors)

            # --- Get wrist pose in elbow frame ---
            wrist_in_elbow = np.dot(
                np.linalg.inv(T_torso_elbow), [wrist_position[0], wrist_position[1], wrist_position[2], 1.0]
            )
            self.markers.markers.append(
                self.create_sphere(self.apply_offset(wrist_in_elbow[0:3]), ColorRGBA(r=0.9, g=0.3, b=0.3, a=1.0), 0.03)
            )

            # [0.0, -math.asin(wrist_in_elbow[2]/d_elbow_wrist), math.atan2(wrist_in_elbow[1], wrist_in_elbow[0])]

            # Get elbow roll
            alpha_elbow = -np.pi / 2 + math.atan2(wrist_in_elbow[2], -wrist_in_elbow[1])
            # print(math.atan2(wrist_in_elbow[2], -wrist_in_elbow[1]))

            rotation_y = R.from_euler("xyz", [-alpha_elbow, 0.0, 0.0])
            Tx = self.make_transformation_matrix([0.0, 0.0, 0.0], [-alpha_elbow, 0.0, 0.0])
            T_shoulder_elbow = np.dot(T_torso_elbow, Tx)
            test_pos, test_rot = self.get_position_orientation_from_transformation(np.dot(T_shoulder_elbow, T_test2))
            self.show_frame(test_pos, test_rot, colors_test)

            # --- Get wrist pose in new elbow frame ---
            wrist_in_elbow_bis = np.dot(
                np.linalg.inv(T_shoulder_elbow), [wrist_position[0], wrist_position[1], wrist_position[2], 1.0]
            )
            print("wrist_in_elbow_bis ", wrist_in_elbow_bis)
            self.markers.markers.append(
                self.create_sphere(self.apply_offset(wrist_in_elbow_bis[0:3]), ColorRGBA(r=0.9, g=0.3, b=0.3, a=1.0), 0.03)
            )

            # print(np.arcsin(wrist_in_elbow_bis[2]/np.sqrt(wrist_in_elbow_bis[0]**2 + wrist_in_elbow_bis[2]**2)))

            # Get elbow pitch
            beta_elbow = -np.arcsin(wrist_in_elbow_bis[2] / np.sqrt(wrist_in_elbow_bis[0] ** 2 + wrist_in_elbow_bis[2] ** 2))
            if wrist_in_elbow_bis[0] < 0:
                beta_elbow = np.pi - beta_elbow

            # print(np.arctan2(wrist_in_elbow_bis[2], wrist_in_elbow_bis[0]))
            # print(np.pi - np.arccos(wrist_in_elbow[2]/np.sqrt(wrist_in_elbow[1]**2 + wrist_in_elbow[2]**2)))
            # beta_elbow = -math.acos(wrist_in_elbow[0]/(np.sqrt(wrist_in_elbow[0]**2 + wrist_in_elbow[2]**2)))
            # beta_elbow = -beta_elbow

            print("alpha ", alpha_elbow)
            print("beta ", beta_elbow)

            # Code magique pour le coude : modifier les angles alpha et beta du coude permet d'obtenir les mêmes coordonnées du poignet que dans meshcat

            # beta_elbow =-1.5
            roll_axe = R.from_euler("xyz", [0.0, 90.0, 0.0], degrees=True)
            rot_y = R.from_euler("xyz", [-alpha_elbow, 0.0, 0.0])
            rot_x = R.from_euler("xyz", [0.0, beta_elbow, 0.0])
            orientation = orientation * rot_y

            colors = [
                ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
            ]
            self.show_frame(elbow_position, orientation.as_euler("xyz"), colors)
            orientation = orientation * rot_x
            shoulder_pos = orientation.apply([0.28, 0.0, 0.0])
            self.markers.markers.append(
                self.create_sphere(
                    self.apply(self.apply_offset(shoulder_pos), elbox_position), ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0), 0.03
                )
            )

            shoulder_orientation = orientation.as_euler("xyz")
            self.markers.markers.append(
                self.create_arrow(
                    self.apply_offset(elbox_position), shoulder_orientation, ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), 0.28, []
                )
            )

            # --------------------------------------------------------

            # self.show_frame(wrist_position, orientation.as_euler('xyz'), colors)
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

            # test_pos, test_rot = self.get_position_orientation_from_transformation(np.dot(T_torso_wrist, T_test2))
            # self.show_frame(test_pos, test_rot, colors_test)
            # test_pos, test_rot = self.get_position_orientation_from_transformation(np.dot(T_torso_wrist, T_test))
            # self.show_frame(test_pos, test_rot, colors_test)
            # position_wrist = np.dot(T_torso_wrist, position)
            # print(position_wrist)
            # self.markers.markers.append(self.create_sphere(self.apply_offset(position_wrist[0:3]), ColorRGBA(r=0.0, g=0., b=0.0, a=1.), 0.03))
            colors = [
                ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),
                ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),
            ]
            # self.show_frame(wrist_position, orientation.as_euler('xyz'), colors)

            # T = self.make_transformation_matrix(tip_position, goal_orientation)
            # tip2 = np.dot(np.linalg.inv(T_torso_wrist), T)
            # print("tip2 ", tip2)
            # tip_orientation = self.get_scipy_matrix_from_transformation(tip2)

            # --- Get tip pose in wrist frame ---
            tip_in_wrist = np.dot(np.linalg.inv(T_torso_wrist), [tip_position[0], tip_position[1], tip_position[2], 1.0])
            # d_tip_wrist = np.sqrt(tip_in_wrist[0]**2 + tip_in_wrist[1]**2 + tip_in_wrist[2]**2)
            # print("tip ", tip_in_wrist)
            self.markers.markers.append(
                self.create_sphere(self.apply_offset(tip_in_wrist[0:3]), ColorRGBA(r=0.9, g=0.3, b=0.3, a=1.0), 0.03)
            )

            # orientation2 = orientation * tip_orientation
            # self.show_frame(wrist_position, orientation2.as_euler('xyz'), colors)

            colors = [
                ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
            ]
            self.show_frame(tip_position, goal_orientation, colors)
            # self.show_frame(tip_position, [goal_orientation[0], goal_orientation[1], 0.], colors)

            # print(tip_orientation.as_euler('zyx'))
            # # [alpha_wrist, beta_wrist, gamma_wrist] = tip_orientation.as_euler('yzx')
            # print(orientation.as_euler('xyz'))
            # print(math.atan2(tip_in_wrist[1], -tip_in_wrist[0]))
            # beta_wrist = math.asin(tip_in_wrist[1]/d_tip_wrist)

            # Get wrist yaw
            beta_wrist = np.pi - math.atan2(tip_in_wrist[1], -tip_in_wrist[0])
            print(beta_wrist)
            # beta_wrist = 0
            # --- Get wrist pose in new wrist frame ---
            Ty = self.make_transformation_matrix([0.0, 0.0, 0.0], [0.0, 0.0, beta_wrist])
            T_torso_wrist = np.dot(T_torso_wrist, Ty)
            test_pos, test_rot = self.get_position_orientation_from_transformation(np.dot(T_torso_wrist, T_test2))
            # self.show_frame(test_pos, test_rot, colors_test)
            rot_y = R.from_euler("xyz", [0.0, 0.0, beta_wrist])

            wrist_in_elbow_bis = np.dot(np.linalg.inv(T_torso_wrist), [tip_position[0], tip_position[1], tip_position[2], 1.0])

            # Get wrist pitch
            # TODO tester cas avec changements de signes
            alpha_wrist = np.arcsin(wrist_in_elbow_bis[2] / np.sqrt(wrist_in_elbow_bis[0] ** 2 + wrist_in_elbow_bis[2] ** 2))
            # alpha_wrist = 0

            # --- Code magique pour le poignet : modifier les angles alpha et beta du poignet permet d'obtenir les mêmes coordonnées du poignet que dans meshcat

            # alpha_wrist = 0
            # beta_wrist = 0
            # gamma_wrist = 0.

            rot_x = R.from_euler("xyz", [0.0, -alpha_wrist, 0.0])
            rot_y = R.from_euler("xyz", [0.0, 0.0, beta_wrist])
            orientation = orientation * rot_y
            orientation = orientation * rot_x

            Ty = self.make_transformation_matrix([0.0, 0.0, 0.0], [0.0, -alpha_wrist, 0.0])
            T_torso_wrist = np.dot(T_torso_wrist, Ty)
            test_pos, test_rot = self.get_position_orientation_from_transformation(np.dot(T_torso_wrist, T_test2))
            self.show_frame(test_pos, test_rot, colors_test)
            T_torso_tip = np.copy(T_torso_wrist)
            T_torso_tip[0][3] = tip_position[0]
            T_torso_tip[1][3] = tip_position[1]
            T_torso_tip[2][3] = tip_position[2]
            test_pos, test_rot = self.get_position_orientation_from_transformation(np.dot(T_torso_tip, T_test2))
            self.show_frame(test_pos, test_rot, colors_test)

            # TODO : correct gamma_wrist

            mrot = R.from_euler("xyz", goal_orientation)
            wrist_pos = mrot.apply([0.1, 0.0, 0.0])
            wrist_pos = [wrist_pos[0] + goal_position[0], wrist_pos[1] + goal_position[1], wrist_pos[2] + goal_position[2]]
            self.markers.markers.append(
                self.create_sphere(
                    self.apply(self.apply_offset(wrist_pos), tip_position), ColorRGBA(r=1.0, g=0.3, b=0.3, a=1.0), 0.03
                )
            )

            # test= np.dot((T_torso_tip), [0.,0.,0.1,1])
            # self.markers.markers.append(self.create_sphere(self.apply_offset(test[0:3]), ColorRGBA(r=0.9, g=0.3, b=0.3, a=1.), 0.03))

            x_in_wrist = np.dot(np.linalg.inv(T_torso_tip), [wrist_pos[0], wrist_pos[1], wrist_pos[2], 1.0])
            self.markers.markers.append(
                self.create_sphere(self.apply_offset(x_in_wrist[0:3]), ColorRGBA(r=1.0, g=0.3, b=0.3, a=1.0), 0.03)
            )

            gamma_wrist = -orientation.as_euler("xyz")[0]  # + R.from_euler('xyz', goal_orientation).as_euler('xyz')[0]
            gamma_wrist = -math.atan2(x_in_wrist[1], x_in_wrist[2])
            # gamma_wrist = np.pi/2 + math.asin(-x_in_wrist[2]/0.1)
            print(math.asin(-x_in_wrist[2] / 0.1) + np.pi / 2)
            print(math.acos(x_in_wrist[1] / 0.1) + np.pi / 2)
            print(-math.atan2(x_in_wrist[1], x_in_wrist[2]))
            # gamma_wrist = 0
            rot_z = R.from_euler("xyz", [gamma_wrist, 0.0, 0.0])
            orientation = orientation * rot_z

            shoulder_pos = orientation.apply([GRIPPER_SIZE, 0.0, 0.0])
            self.markers.markers.append(
                self.create_sphere(
                    self.apply(self.apply_offset(shoulder_pos), wrist_position), ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0), 0.03
                )
            )
            print("shoulder_pos ", self.apply(shoulder_pos, wrist_position))

            shoulder_orientation = orientation.as_euler("xyz")
            colors = [
                ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),
                ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
                ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),
            ]
            self.show_frame(tip_position, orientation.as_euler("xyz"), colors)
            self.markers.markers.append(
                self.create_arrow(
                    self.apply_offset(wrist_position), shoulder_orientation, ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), 0.15, []
                )
            )
            print(orientation.as_euler("xyz"))

            # --------------------------------------------------------

            print("Data : ")
            print(
                "meshcat joints", [alpha_shoulder, beta_shoulder, alpha_elbow, beta_elbow, alpha_wrist, beta_wrist, gamma_wrist]
            )
            print("shoulder : ", [0.0, -0.2, 0.0], [0.0, 0.0, 0.0])
            print("elbow : ", elbox_position)
            print("wrist : ", self.get_wrist_position(goal_pose))
            print("tip : ", goal_pose[0], goal_pose[1])

            print("Done")

        self.marker_pub.publish(self.markers)

    def get_scipy_matrix_from_transformation(self, T):
        return R.from_matrix(T[0:3, 0:3])

    def get_position_orientation_from_transformation(self, T_torso_wrist):
        rotation = [[T_torso_wrist[0][0:3], T_torso_wrist[1][0:3], T_torso_wrist[2][0:3]]]
        rotation = (R.from_matrix(rotation)).as_euler("xyz")[0]
        position = [T_torso_wrist[0][3], T_torso_wrist[1][3], T_torso_wrist[2][3]]
        return position, rotation

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

    def apply(self, position, offsets):
        return [position[0] + offsets[0], position[1] + offsets[1], position[2] + offsets[2]]

    def apply_shoulder(self, position, offset=-0.2):
        return [position[0], position[1] + offset, position[2]]

    def go_to_position(self, joint_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], wait=10):
        """
        Show pose with the r_arm in meshcat

        args:
            joint_pose: joint pose of the arm
            wait: time to wait before closing the window
        """
        # self.ik_reachy.setup(urdf_path="../reachy_placo/reachy_placo/reachy/new_new_reachy2_1.urdf")
        names = self.r_arm_joint_names()
        for i in range(len(names)):
            self.ik_reachy.robot.set_joint(names[i], joint_pose[i])
        self.ik_reachy._tick_viewer()

        print(self.ik_reachy.robot.get_T_a_b("torso", "r_elbow_yaw"))
        print(self.ik_reachy.robot.get_T_a_b("r_shoulder_roll", "r_elbow_yaw"))
        print(self.ik_reachy.robot.get_T_a_b("torso", "r_tip_joint"))
        print(self.ik_reachy.robot.get_T_a_b("r_wrist_roll", "r_tip_joint"))

        time.sleep(wait)

    def show_frame(self, position, orientation, colors):
        roll_axe = R.from_euler("xyz", orientation)
        pitch_axe = roll_axe * R.from_euler("xyz", [0.0, 0.0, math.pi / 2])
        yaw_axe = roll_axe * R.from_euler("xyz", [0.0, -math.pi / 2, 0.0])
        roll_axe = roll_axe.as_euler("xyz")
        pitch_axe = pitch_axe.as_euler("xyz")
        yaw_axe = yaw_axe.as_euler("xyz")
        self.markers.markers.append(self.create_arrow(self.apply_offset(position), roll_axe, colors[0], 0.1, []))
        self.markers.markers.append(self.create_arrow(self.apply_offset(position), pitch_axe, colors[1], 0.1, []))
        self.markers.markers.append(self.create_arrow(self.apply_offset(position), yaw_axe, colors[2], 0.1, []))

    def apply_offset(self, position):
        return [position[0] + X_OFFSET, position[1], position[2]]

    def get_wrist_position(self, goal_pose):
        Mrot = R.from_euler("xyz", goal_pose[1])
        wrist_pos = Mrot.apply([0.0, 0.0, GRIPPER_SIZE])
        wrist_pos = [wrist_pos[0] + goal_pose[0][0], wrist_pos[1] + goal_pose[0][1], wrist_pos[2] + goal_pose[0][2]]
        return wrist_pos
        # print("wrist_pos ", wrist_pos)

    def show_robot_arm(self, goal_pose):
        self.markers.markers.append(
            self.create_sphere(self.apply_offset([0.0, 0.0, 0.0]), ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), 0.05)
        )
        self.markers.markers.append(
            self.create_sphere(self.apply_offset([0.0, -0.2, 0.0]), ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), 0.03)
        )
        wrist_position = self.get_wrist_position(goal_pose)
        print("wrist_position ", wrist_position)
        self.markers.markers.append(
            self.create_sphere(self.apply_offset(wrist_position), ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0), 0.03)
        )
        print("wrists ", wrist_position)
        self.markers.markers.append(
            self.create_sphere(self.apply_offset([0.0, -0.2, 0.0]), ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.2), UPPER_ARM_SIZE * 2)
        )
        self.markers.markers.append(
            self.create_sphere(self.apply_offset(wrist_position), ColorRGBA(r=1.0, g=0.0, b=0.5, a=0.2), FOREARM_SIZE * 2)
        )

    def show_goal_position(self, goal_pose):
        self.markers.markers.append(
            self.create_sphere(
                [goal_pose[0][0] + 3, goal_pose[0][1], goal_pose[0][2]], ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0), 0.03
            )
        )
        self.markers.markers.append(
            self.create_arrow(
                [goal_pose[0][0] + 3, goal_pose[0][1], goal_pose[0][2]],
                goal_pose[1],
                ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),
                0.1,
                [],
            )
        )

    def get_coordinate_cercle(self, d, theta):
        a = 1 / (2 * d) * np.sqrt(4 * d**2 * UPPER_ARM_SIZE**2 - (d**2 - FOREARM_SIZE**2 + UPPER_ARM_SIZE**2) ** 2)
        x = (d**2 - FOREARM_SIZE**2 + UPPER_ARM_SIZE**2) / (2 * d)
        y = -a * np.cos(theta)
        z = a * np.sin(theta)
        return [x, y, z]

    def change_frame(self, xyz):
        print("xyz ", xyz)
        d = math.sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2])
        dproj = math.sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1])
        print("d ", dproj)
        if d == 0:
            # return identity matrix 3x3
            return np.identity(3)
        a = xyz[0] / dproj
        b = xyz[1] / dproj
        c = xyz[2] / d
        e = dproj / d
        M = np.array([[a, -b, 0.0], [b, a, 0], [0, 0, 1]])
        m2 = np.array([[e, 0, c], [0, 1, 0], [-c, 0, e]])

        return M, m2

    def create_sphere(self, xyz, color, size):
        ros_time = self.get_clock().now()
        msg = Marker()
        msg.id = self.id
        self.id = self.id + 1
        msg.frame_locked = True
        msg.action = Marker.ADD
        msg.header.frame_id = "torso"
        msg.header.stamp = ros_time.to_msg()
        msg.type = Marker.SPHERE
        msg.mesh_use_embedded_materials = False
        msg.scale = Vector3(x=size, y=size, z=size)
        msg.color = color
        # msg.pose.position = MESH_POSITION
        msg.ns = "Position"
        msg.lifetime.sec = 0
        msg.lifetime.nanosec = 0
        msg.pose.position.x = xyz[0]
        msg.pose.position.y = xyz[1]
        msg.pose.position.z = xyz[2]
        return msg

    def create_arrow(self, xyz, rpy, color, size, q):
        ros_time = self.get_clock().now()
        msg = Marker()
        msg.id = self.id
        self.id = self.id + 1
        msg.frame_locked = True
        msg.action = Marker.ADD
        msg.header.frame_id = "torso"
        msg.header.stamp = ros_time.to_msg()
        msg.type = Marker.ARROW
        msg.mesh_use_embedded_materials = False
        msg.scale = Vector3(x=size, y=0.01, z=0.01)
        msg.color = color
        # msg.pose.position = MESH_POSITION
        msg.ns = "Position"
        msg.lifetime.sec = 0
        msg.lifetime.nanosec = 0
        msg.pose.position.x = xyz[0]
        msg.pose.position.y = xyz[1]
        msg.pose.position.z = xyz[2]
        if len(q) == 0:
            q = tf_transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
            # r,p,y = tf_transformations.euler_from_quaternion(q)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        return msg


def main(args=None):
    rclpy.init(args=args)
    arrow_publisher = ArrowPublisher()
    rclpy.spin(arrow_publisher)
    arrow_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
