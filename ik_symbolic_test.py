from ik_symbolic import IK_symbolic
import numpy as np
from reachy_placo.ik_reachy_placo import IKReachyQP
import math
import time
from scipy.spatial.transform import Rotation as R
from grasping_utils.utils import (
    get_pose_msg_from_euler,
    get_homogeneous_matrix_msg_from_euler,
    get_pose_msg_from_homogeneous_matrix,
)


class IKSymbolicTest:
    def __init__(self):
        print("--- IK Symbolic Test ---")
        self.ik = IK_symbolic()
        self.ik_reachy = IKReachyQP(
            viewer_on=True,
            collision_avoidance=False,
            parts=["r_arm"],
            position_weight=1.9,
            orientation_weight=1e-2,
            robot_version="reachy_2",
            velocity_limit=50.0,
        )
        self.ik_reachy.setup(urdf_path="../reachy_placo/reachy_placo/reachy/new_new_reachy2_2.urdf")
        self.ik_reachy.create_tasks()
        self.green = "\033[92m"  # GREEN
        self.blue = "\033[94m"  # BLUE
        self.yellow = "\033[93m"  # YELLOW
        self.red = "\033[91m"  # RED
        self.reset_color = "\033[0m"  # RESET COLOR

    def time_test(self):
        # TODO : tester le temps de calcul
        # fail test
        goal_pose = [[0.1, -0.2, 0.1], [30, -20, -10]]
        start = time.time()
        for i in range(1000):
            self.ik.is_reachable(goal_pose)
        end = time.time()
        print("time fail : ", end - start)

        # success test
        goal_pose = [[0.2, -0.2, -0.1], [30, -20, -10]]
        start = time.time()
        for i in range(1000):
            self.ik.is_reachable(goal_pose)
        end = time.time()
        print("time success: ", end - start)

        goal_pose = [[0.2, -0.2, -0.1], [30, -20, -10]]
        result = self.ik.is_reachable(goal_pose)
        start = time.time()
        for i in range(1000):
            result[2](limits[0])
        end = time.time()
        print("time shoulder : ", end - start)

        return 0

    def joints_space_test(self):
        placo_success = 0
        placo_fail = 0
        symbolic_success = 0
        symbolic_fail = 0
        print("succes test")
        for k in range(100):
            print("--------------------")
            shoulder_pitch = np.random.uniform(-math.pi, math.pi)
            shoulder_roll = np.random.uniform(-math.pi, math.pi)
            elbow_yaw = np.random.uniform(-math.pi, math.pi)
            elbow_pitch = np.random.uniform(-math.pi, math.pi)
            wrist_pitch = np.random.uniform(-math.pi / 4, math.pi / 4)
            wrist_roll = np.random.uniform(-math.pi / 4, math.pi / 4)
            wrist_yaw = np.random.uniform(-math.pi, math.pi)
            joints = [shoulder_pitch, shoulder_roll, elbow_yaw, elbow_pitch, wrist_pitch, wrist_roll, wrist_yaw]
            if (
                (shoulder_roll < np.radians(-150))
                or (shoulder_roll > np.radians(40))
                or (elbow_pitch < np.radians(-130))
                or (elbow_pitch > np.radians(0))
            ):
                print(self.yellow + str(np.degrees(joints)) + self.reset_color)
            else:
                print(self.blue + str(np.degrees(joints)) + self.reset_color)
            self.go_to_position(joints, wait=0.0)
            goal_pose_matrix = self.ik_reachy.robot.get_T_a_b("torso", "r_tip_joint")
            goal_pose[0] = goal_pose_matrix[:3, 3]
            goal_pose[1] = R.from_matrix(goal_pose_matrix[:3, :3]).as_euler("xyz")
            is_reachable, joints_placo, errors = self.ik_reachy.is_pose_reachable(
                goal_pose_matrix,
                arm_name="r_arm",
                q0=[0.0, 0.0, 0.0, -math.pi / 2, 0, 0.0, 0.0],
                tolerances=[0.001, 0.001, 0.001, 0.02, 0.02, 0.02],
                max_iter=45,
                nb_stepper_solve=25,
            )
            if is_reachable:
                print(self.green + "Placo reachable" + self.reset_color)
                self.go_to_position(joints_placo, wait=5.0)
                placo_success += 1
            else:
                print(self.red + "Placo not reachable" + self.reset_color)
            result = self.ik.is_reachable(goal_pose)
            if result[0]:
                print(self.green + "Symbolic reachable" + self.reset_color)
                symbolic_success += 1
            else:
                print(self.red + "Symbolic not reachable" + self.reset_color)
            time.sleep(0.5)
        print("fail test")
        for k in range(100):
            print("--------------------")
            shoulder_pitch = np.random.uniform(-math.pi, math.pi)
            shoulder_roll = np.random.uniform(-math.pi, math.pi)
            elbow_yaw = np.random.uniform(-math.pi, math.pi)
            elbow_pitch = np.random.uniform(-math.pi, math.pi)
            wrist_pitch = np.random.uniform(math.pi / 4, 2 * math.pi - math.pi / 4)
            wrist_roll = np.random.uniform(math.pi / 4, 2 * math.pi - math.pi / 4)
            wrist_yaw = np.random.uniform(-math.pi, math.pi)
            joints = [shoulder_pitch, shoulder_roll, elbow_yaw, elbow_pitch, wrist_pitch, wrist_roll, wrist_yaw]
            print(self.blue + str(np.degrees(joints)) + self.reset_color)
            self.go_to_position(joints, wait=0.0)
            goal_pose_matrix = self.ik_reachy.robot.get_T_a_b("torso", "r_tip_joint")
            goal_pose[0] = goal_pose_matrix[:3, 3]
            goal_pose[1] = R.from_matrix(goal_pose_matrix[:3, :3]).as_euler("xyz")
            is_reachable, joints_placo, errors = self.ik_reachy.is_pose_reachable(
                goal_pose_matrix,
                arm_name="r_arm",
                q0=[0.0, 0.0, 0.0, -math.pi / 2, 0, 0.0, 0.0],
                tolerances=[0.001, 0.001, 0.001, 0.02, 0.02, 0.02],
                max_iter=45,
                nb_stepper_solve=25,
            )
            if is_reachable:
                print(self.green + "Placo reachable" + self.reset_color)
                placo_fail += 1
                # self.go_to_position(joints_placo, wait=5.0)
            else:
                print(self.red + "Placo not reachable" + self.reset_color)
            result = self.ik.is_reachable(goal_pose)
            if result[0]:
                print(self.green + "Symbolic reachable" + self.reset_color)
                symbolic_fail += 1
            else:
                print(self.red + "Symbolic not reachable" + self.reset_color)
            time.sleep(0.5)
        print("Placo success : ", placo_success)
        print("Symbolic success : ", symbolic_success)
        print("Placo fail : ", placo_fail)
        print("Symbolic fail : ", symbolic_fail)

    def task_space_test(self):
        # TODO : tester une pose atteignable par Placo et qu'on peut obtenir la meme solution
        return 0

    def make_movement_test(self, goal_pose):
        # TODO : tester tous les theta du cercle des solutions
        # print("test")
        result = self.ik.is_reachable(goal_pose)
        # print("tessst")
        if result[0]:
            print(int((result[1][1] - result[1][0]) * 100))
            angles = np.linspace(result[1][0], result[1][1], int((result[1][1] - result[1][0]) * 50))
            while True:
                # print("eee")
                # print("e", angles)
                for angle in angles:
                    # print("limits : ", result[1])
                    joints = result[2](angle)
                    # print(joints)
                    self.go_to_position(joints, wait=0.0)
                    # time.sleep(0.1)
        else:
            print("Pose not reachable")

    def good_poses_test(self, goal_pose, angle_precision=0.5):
        # TODO : tester une pose atteignable par Placo et qu'on peut obtenir la meme solution
        goal_pose_matrix = get_homogeneous_matrix_msg_from_euler(goal_pose[0], goal_pose[1])
        is_reachable, joints_placo, errors = self.ik_reachy.is_pose_reachable(
            goal_pose_matrix,
            arm_name="r_arm",
            q0=[0.0, 0.0, 0.0, -math.pi / 2, 0, 0.0, 0.0],
            tolerances=[0.001, 0.001, 0.001, 0.02, 0.02, 0.02],
            max_iter=45,
            nb_stepper_solve=25,
        )
        self.go_to_position(joints_placo, wait=5.0)
        if is_reachable:
            result = self.ik.is_reachable(goal_pose)
            if result[0]:
                thetas = np.linspace(result[1][0], result[1][1], int((result[1][1] - result[1][0]) * 50))
                print(len(thetas))
                for theta in thetas:
                    # print("theta : ", theta)
                    joints_symbolic = result[2](theta)
                    joints_placo = np.array(joints_placo)
                    joints_symbolic = np.array(joints_symbolic)
                    if np.all(np.abs(joints_symbolic - joints_placo) < np.deg2rad(angle_precision)):
                        print("Same solution for theta : ", theta)
                        joints_success = joints_symbolic
                print("The end")
                self.go_to_position(joints_success, wait=5.0)
            else:
                print("Not reachable by symbolic")
        else:
            print("Not reachable by Placo")

    def show_in_meshcat(self, joints):
        # TODO : montre un resultat dans meshcat
        return 0

    def are_joints_correct(self, joints, goal_pose, position_tolerance=0.0001, orientation_tolerance=0.0001):
        self.go_to_position(joints, wait=0)
        T_torso_tip = self.ik_reachy.robot.get_T_a_b("torso", "r_tip_joint")
        position = T_torso_tip[:3, 3]
        orientation = T_torso_tip[:3, :3]
        orientation = R.from_matrix(orientation).as_euler("xyz")

        print("goal_orientation : ", goal_pose[1])
        print("Position : ", position)
        print("Orientation : ", orientation)
        print("Goal Position : ", goal_position)
        print("Goal Orientation : ", goal_pose[1])
        for i in range(3):
            if abs(position[i] - goal_position[i]) > position_tolerance:
                print("Position not correct")
                return False
        for i in range(3):
            if abs(orientation[i] - goal_orientation[i]) > orientation_tolerance:
                print("Orientation not correct")
                return False
        return True

    def go_to_position(self, joint_pose=[0.0, 0.0, 0.0, -math.pi / 2, 0.0, 0.0, 0.0], wait=10):
        """
        Show pose with the r_arm in meshcat

        args:
            joint_pose: joint pose of the arm
            wait: time to wait before closing the window
        """
        # self.ik_reachy.setup(urdf_path="../reachy_placo/reachy_placo/reachy/new_new_reachy2_2.urdf")
        names = self.r_arm_joint_names()
        for i in range(len(names)):
            self.ik_reachy.robot.set_joint(names[i], joint_pose[i])
        self.ik_reachy._tick_viewer()

        # print(self.ik_reachy.robot.get_T_a_b("torso", "r_elbow_yaw"))
        # print(self.ik_reachy.robot.get_T_a_b("r_shoulder_roll", "r_elbow_yaw"))
        # print(self.ik_reachy.robot.get_T_a_b("torso", "r_tip_joint"))
        # print(self.ik_reachy.robot.get_T_a_b("r_wrist_roll", "r_tip_joint"))

        time.sleep(wait)

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

    def r_arm_joint_names(self):
        names = []
        names.append("r_shoulder_pitch")
        names.append("r_shoulder_roll")
        names.append("r_elbow_yaw")
        names.append("r_elbow_pitch")
        names.append("r_wrist_roll")
        names.append("r_wrist_pitch")
        names.append("r_wrist_yaw")
        return names


if __name__ == "__main__":
    ik_symbolic_test = IKSymbolicTest()

    goal_position = [0.4, -0.2, -0.1]
    goal_orientation = [30, -50, 20]
    goal_orientation = np.deg2rad(goal_orientation)
    goal_pose = [goal_position, goal_orientation]

    [is_reachable, limits, get_joints] = ik_symbolic_test.ik.is_reachable(goal_pose)
    # ik_symbolic_test.go_to_position(get_joints(limits[0]), wait=8)
    # print(is_reachable)
    # print(is_reachable, limits)
    # if is_reachable:
    #     joints = get_joints(np.pi)
    #     print(joints)
    #     is_correct = ik_symbolic_test.are_joints_correct(joints, goal_pose)
    #     print(is_correct)
    #     ik_symbolic_test.go_to_position(joints, wait=8)
    # else:
    #     print("Pose not reachable")

    # ik_symbolic_test.go_to_position([0.0,0.0, 0.0, -math.pi / 2, 0.0, 0.0, 0.0], wait=8)
    # print(ik_symbolic_test.ik_reachy.robot.get_T_a_b("torso", "r_tip_joint"))

    # goal_position = [0.2, -0.2, -0.1]
    # goal_orientation = [-0,-90,0]
    # goal_orientation = np.deg2rad(goal_orientation)
    # goal_pose = [goal_position, goal_orientation]
    # ik_symbolic_test.make_movement_test(goal_pose)

    # ik_symbolic_test.time_test()
    # ik_symbolic_test.joints_space_test()
    # ik_symbolic_test.task_space_test()
    ik_symbolic_test.good_poses_test(goal_pose)
