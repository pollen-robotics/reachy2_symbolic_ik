from reachy2_symbolic_ik.ik_symbolic import IK_symbolic
import numpy as np
from reachy_placo.ik_reachy_placo import IKReachyQP
import math
import time
from scipy.spatial.transform import Rotation as R
from grasping_utils.utils import get_homogeneous_matrix_msg_from_euler


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
        self.ik_reachy.setup(urdf_path="../reachy_placo/reachy_placo/reachy/new_new_reachy2_1.urdf")
        self.ik_reachy.create_tasks()
        self.green = "\033[92m"  # GREEN
        self.blue = "\033[94m"  # BLUE
        self.yellow = "\033[93m"  # YELLOW
        self.red = "\033[91m"  # RED
        self.reset_color = "\033[0m"  # RESET COLOR

    def make_line(self, start_position, end_position, start_orientation, end_orientation, nb_points=100):
        x = np.linspace(start_position[0], end_position[0], nb_points)
        y = np.linspace(start_position[1], end_position[1], nb_points)
        z = np.linspace(start_position[2], end_position[2], nb_points)
        roll = np.linspace(start_orientation[0], end_orientation[0], nb_points)
        pitch = np.linspace(start_orientation[1], end_orientation[1], nb_points)
        yaw = np.linspace(start_orientation[2], end_orientation[2], nb_points)
        time.sleep(5)
        for i in range(nb_points):
            goal_pose = [[x[i], y[i], z[i]], [roll[i], pitch[i], yaw[i]]]
            result = self.ik.is_reachable(goal_pose)

            if result[0]:
                angle = np.linspace(result[1][0], result[1][1], 3)[1]
                # print(angle)
                joints = result[2](angle)
                # print(result[1])
                print(result[1])
                self.go_to_position(joints, wait=0.0)
                # time.sleep(0.1)
            else:
                print("Pose not reachable")

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
            result[2](result[1][0])
        end = time.time()
        print("time shoulder : ", end - start)

    def joints_space_test(self):
        placo_success = 0
        placo_fail = 0
        symbolic_success = 0
        symbolic_fail = 0
        print("succes test")
        pose_fail = []
        goal_pose = [[], []]
        test = 0
        for k in range(1000):
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
            # print("goal_pose : ", goal_pose)
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
                # self.go_to_position(joints_placo, wait=5.0)
                placo_success += 1
            else:
                print(self.red + "Placo not reachable" + self.reset_color)
            result = self.ik.is_reachable(goal_pose)
            if result[0]:
                print(self.green + "Symbolic reachable" + self.reset_color)
                symbolic_success += 1
                result[2](result[1][0])
                result[2](result[1][1])
                result[2]((result[1][0] + result[1][1]) / 2)
                joints = result[2](result[1][0])
                joints2 = self.ik.get_joints2(result[1][0])
                if (
                    (np.abs(joints[0] - joints2[0]) < 0.00001)
                    and (np.abs(joints[1] - joints2[1]) < 0.00001)
                    and (np.abs(joints[2] - joints2[2]) < 0.00001)
                    and (np.abs(joints[3] - joints2[3]) < 0.00001)
                    and (np.abs(joints[4] - joints2[4]) < 0.00001)
                    and (np.abs(joints[5] - joints2[5]) < 0.00001)
                    and (np.abs(joints[6] - joints2[6]) < 0.00001)
                ):
                    test += 1
                # print(np.abs(joints[0]-joints2[0])<0.00001)
                # print(np.abs(joints[1]-joints2[1])<0.00001)
                # print("joints : ", joints)
                # print("joints2 : ", joints2)

            else:
                print(self.red + "Symbolic not reachable" + self.reset_color)
                print("goal_pose : ", goal_pose)
                pose_fail.append(np.copy(goal_pose))
                print("pose fail : ", pose_fail)

            # time.sleep(0.5)
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
            # time.sleep(0.5)
        print("Pose fail : ", pose_fail)
        print("Placo success : ", placo_success)
        print("Symbolic success : ", symbolic_success)
        print("Placo fail : ", placo_fail)
        print("Symbolic fail : ", symbolic_fail)
        print("test : ", test)

    def task_space_test(self, x_step=0.15, y_step=0.15, z_step=0.15, roll_step=45, pitch_step=45, yaw_step=45):
        # TODO : tester une pose atteignable par Placo et qu'on peut obtenir la meme solution
        shoulder_position = self.ik.shoulder_position
        arm_length = self.ik.upper_arm_size + self.ik.forearm_size + self.ik.gripper_size
        arm_length = 0.5
        goal_poses = []
        reachable_poses = 0
        start_time = time.time()
        for x in np.arange(shoulder_position[0] - arm_length, shoulder_position[0] + arm_length + x_step, x_step):
            for y in np.arange(shoulder_position[1] - arm_length, shoulder_position[1] + arm_length + y_step, y_step):
                for z in np.arange(shoulder_position[2] - arm_length, shoulder_position[2] + arm_length + z_step, z_step):
                    goal_position = (x, y, z)
                    # verify if the position is in the sphere of the arm and is if it's only in front of the robot
                    if (np.linalg.norm(np.array(goal_position) - np.array(shoulder_position)) > arm_length) or (goal_position[0] < 0):
                        continue
                    for roll in np.arange(0, 360, roll_step):
                        for pitch in np.arange(0, 360, pitch_step):
                            for yaw in np.arange(0, 360, yaw_step):
                                goal_orientation = (roll, pitch, yaw)
                                goal_orientation = np.deg2rad(goal_orientation)
                                goal_pose = [goal_position, goal_orientation]
                                goal_poses.append(goal_pose)
        end_time = time.time()
        print("time : ", end_time - start_time)
        print("goal_poses : ", len(goal_poses))

        start_time = time.time()
        for i in range(len(goal_poses)):
            # print(i)
            result = self.ik.is_reachable(goal_poses[i])
            if result[0]:
                reachable_poses += 1
        end_time = time.time()
        print("time : ", end_time - start_time)
        print("reachable poses : ", reachable_poses)
        print("total poses : ", len(goal_poses))


    def make_movement_test(self, goal_pose):
        # TODO : tester tous les theta du cercle des solutions
        # print("test")
        result = self.ik.is_reachable(goal_pose)
        # print("tessst")
        if result[0]:
            print(int((result[1][1] - result[1][0]) * 50))
            angles = np.linspace(result[1][0], result[1][1], int((result[1][1] - result[1][0]) * 50))
            while True:
                # print("eee")
                # print("e", angles)
                for angle in angles:
                    # print("limits : ", result[1])
                    joints = result[2](angle)

                    # print(joints)
                    self.go_to_position(joints, wait=0.1)
                    # time.sleep(0.1)
        else:
            print("Pose not reachable")

    def good_poses_test(self, goal_pose, angle_precision=7):
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
        # self.go_to_position(joints_placo, wait=5.0)
        joints_placo = [(joint + 2 * np.pi) % (2 * np.pi) for joint in joints_placo]
        joints_placo = np.array(joints_placo)
        if is_reachable:
            result = self.ik.is_reachable(goal_pose)
            if result[0]:
                thetas = np.linspace(result[1][0], result[1][1], int((result[1][1] - result[1][0]) * 50))
                print(len(thetas))
                for theta in thetas:
                    # print("theta : ", theta)
                    joints_symbolic = result[2](theta)

                    # self.go_to_position(joints_symbolic, wait=0.1)
                    joints_symbolic = [(joint + 2 * np.pi) % (2 * np.pi) for joint in joints_symbolic]

                    joints_symbolic = np.array(joints_symbolic)
                    # joints_placo = (joints_placo + 2*np.pi) % 2*np.pi;
                    # joints_symbolic = (joints_symbolic + 2*np.pi) % 2*np.pi;

                    if np.all(np.abs(joints_symbolic - joints_placo) < np.deg2rad(angle_precision)):
                        print("joints_placo : ", np.degrees(joints_placo))
                        print("joints_symbolic : ", np.degrees(joints_symbolic))
                        print("Same solution for theta : ", theta, np.degrees(np.abs(joints_symbolic - joints_placo)))
                        # self.go_to_position(joints_symbolic, wait=5.0)
                print("The end")
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
        goal_position = goal_pose[0]
        goal_orientation = goal_pose[1]

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
        # self.ik_reachy.setup(urdf_path="../reachy_placo/reachy_placo/reachy/new_new_reachy2_1.urdf")
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

    # ------ Ik function test ------

    # goal_position = [0.4, 0.1, -0.4]
    # goal_orientation = [20, -80, 10]
    # goal_orientation = np.deg2rad(goal_orientation)
    # goal_pose = [goal_position, goal_orientation]

    # result = ik_symbolic_test.ik.is_reachable(goal_pose)
    # if result[0]:
    #     joints = result[2](result[1][0])
    #     ik_symbolic_test.go_to_position(joints, wait=8)

    # ------ Make line test ------

    # start_position = [0.4, 0.1, -0.4]
    # end_position = [0.3, -0.2, -0.1]
    # start_orientation = np.deg2rad([20, -80, 10])
    # end_orientation = np.deg2rad([0, -0, 0])
    # ik_symbolic_test.make_line(start_position, end_position, start_orientation, end_orientation, nb_points=300)

    # ------ Make movement test ------

    # goal_position = [0.2, -0.2, -0.1]
    # goal_orientation = [-0,-90,0]
    # goal_orientation = np.deg2rad(goal_orientation)
    # goal_pose = [goal_position, goal_orientation]
    # ik_symbolic_test.make_movement_test(goal_pose)

    # ------ Good poses test ------

    # goal_position = [0.2, -0.2, -0.1]
    # goal_orientation = [-0,-90,0]
    # goal_orientation = np.deg2rad(goal_orientation)
    # goal_pose = [goal_position, goal_orientation]
    # ik_symbolic_test.good_poses_test(goal_pose)

    # ------ Are joints correct test ------

    # goal_position = [0.2, -0.2, -0.1]
    # goal_orientation = [-0,-90,0]
    # goal_orientation = np.deg2rad(goal_orientation)
    # goal_pose = [goal_position, goal_orientation]
    # [is_reachable, limits, get_joints] = ik_symbolic_test.ik.is_reachable(goal_pose)
    # if is_reachable:
    #     joints = get_joints(limits[0])
    #     is_correct = ik_symbolic_test.are_joints_correct(joints, goal_pose)
    #     print(is_correct)
    #     ik_symbolic_test.go_to_position(joints, wait=8)
    # else:
    #     print("Pose not reachable")

    # ------ Go to position test ------

    # ik_symbolic_test.go_to_position([0.0,0.0, 0.0, -math.pi / 2, 0.0, 0.0, 0.0], wait=8)
    # print(ik_symbolic_test.ik_reachy.robot.get_T_a_b("torso", "r_tip_joint"))

    # ------ Joints space test ------

    # ik_symbolic_test.joints_space_test()

    # ------ Task space test ------

    ik_symbolic_test.task_space_test()

    # ------ Time test ------

    # ik_symbolic_test.time_test()

    # ------ Other tests ------

    # goal_poses = [[[-0.16451815, -0.4785408 ,  0.00735828], [-0.34020077, -0.22254548, -0.24598186]],
    #    [[ 0.17320635, -0.28877271,  0.12595165], [ 1.91043709,  0.02182434, -0.59891522]],
    #    [[ 0.11657383, -0.31879514, -0.18552353], [-2.06584127, -0.50205104,  0.56725307]],
    #    [[ 0.16831123, -0.12026895, -0.03128163],[-0.71848228,  0.5489851 ,  1.28165872]],
    #    [[ 0.20250884, -0.25782176,  0.29294886], [-0.91518747,  1.10257414,  2.48795061]],
    #    [[-0.26389982, -0.28966803,  0.03440583], [-2.63976015,  0.98274754, -0.90832566]],
    #    [[-0.19330538, -0.50181275,  0.22097992], [-1.05261722,  0.03576907,  0.11900632]],
    #    [[ 0.06531758,  0.03234202, -0.29959564], [-0.66863851,  0.68699396, -0.77430419]],
    #    [[0.19536822, 0.06314997, 0.15404791], [3.05383503, 0.31518798, 2.62346594]],
    #    [[ 0.01433523, -0.22712099, -0.32676204], [-1.14603432, -0.70486436, -2.91049123]],
    #    [[ 0.11424737, -0.02766657,  0.02796033],[-2.34392801,  0.66672375,  0.71425915]],
    #    [[-0.19752591, -0.57261312, -0.06887125], [-0.06028131, -0.34602401, -0.91035576]],
    #    [[-0.04438143,  0.05823801, -0.24362076], [ 2.2769857 ,  0.61654679,  0.85133659]],
    #    [[-0.03828849,  0.12047255,  0.09052939],[ 2.97823218, -0.08923544,  1.40832877]]]

    # for goal_pose in goal_poses:
    #     ik_symbolic_test = IKSymbolicTest()
    #     [is_reachable, limits, get_joints] = ik_symbolic_test.ik.is_reachable(goal_pose)
    #     print(is_reachable)
