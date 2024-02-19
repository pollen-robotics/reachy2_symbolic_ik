import math
import time
from pathlib import Path

import numpy as np
from grasping_utils.utils import get_homogeneous_matrix_msg_from_euler
from reachy_placo.ik_reachy_placo import IKReachyQP
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import shoulder_limits
from reachy2_symbolic_ik.utils_placo import go_to_position


def time_test(symbolic_ik: SymbolicIK, placo_ik: IKReachyQP) -> None:
    print("TIME TEST")
    print("unreachable pose")
    goal_pose = [[0.1, -0.2, 0.1], [30, -20, -10]]
    goal_pose_matrix = get_homogeneous_matrix_msg_from_euler(goal_pose[0], goal_pose[1])
    start = time.time()
    for i in range(1000):
        symbolic_ik.is_reachable(goal_pose)
    end = time.time()
    print("symbolic ik : ", end - start)
    start = time.time()
    for i in range(1000):
        placo_ik.is_pose_reachable(
            goal_pose_matrix,
            arm_name="r_arm",
            q0=[0.0, -math.pi / 2, 0.0, -math.pi / 2, 0, 0.0, 0.0],
            tolerances=[0.001, 0.001, 0.001, 0.02, 0.02, 0.02],
            max_iter=45,
            nb_stepper_solve=25,
        )
    end = time.time()
    print("placo ik : ", end - start)

    print("reachable pose")
    goal_pose = [[0.6, -0.2, -0], [0, -90, 0]]
    goal_pose_matrix = get_homogeneous_matrix_msg_from_euler(goal_pose[0], goal_pose[1])
    start = time.time()
    for i in range(1000):
        symbolic_ik.is_reachable(goal_pose)
    end = time.time()
    print("symbolic ik - is_reachable : ", end - start)
    result = symbolic_ik.is_reachable(goal_pose)
    start = time.time()
    for i in range(1000):
        result[2](result[1][0])
    end = time.time()
    print("symbolic ik - get joints : ", end - start)
    start = time.time()
    for i in range(1000):
        placo_ik.is_pose_reachable(
            goal_pose_matrix,
            arm_name="r_arm",
            q0=[0.0, -math.pi / 2, 0.0, -math.pi / 2, 0, 0.0, 0.0],
            tolerances=[0.001, 0.001, 0.001, 0.02, 0.02, 0.02],
            max_iter=45,
            nb_stepper_solve=25,
        )
    end = time.time()
    print("placo ik : ", end - start)


def joints_space_test(symbolic_ik: SymbolicIK, placo_ik: IKReachyQP, verbose: bool = False, number_of_point: int = 100) -> None:
    green = "\033[92m"  # GREEN
    blue = "\033[94m"  # BLUE
    yellow = "\033[93m"  # YELLOW
    red = "\033[91m"  # RED
    reset_color = "\033[0m"  # RESET COLOR

    placo_success = 0
    symbolic_success = 0
    out_of_limits = 0

    for k in range(number_of_point):
        shoulder_pitch = np.random.uniform(-math.pi, math.pi)
        shoulder_roll = np.random.uniform(-math.pi, math.pi)
        elbow_yaw = np.random.uniform(-math.pi, math.pi)
        elbow_pitch = np.random.uniform(-math.pi, math.pi)
        wrist_pitch = np.random.uniform(-math.pi, math.pi)
        wrist_roll = np.random.uniform(-math.pi / 4, math.pi / 4)
        wrist_yaw = np.random.uniform(-math.pi, math.pi)
        joints = [shoulder_pitch, shoulder_roll, elbow_yaw, elbow_pitch, wrist_pitch, wrist_roll, wrist_yaw]

        if (
            # (shoulder_roll < np.radians(-150))
            # or (shoulder_roll > np.radians(40))
            (elbow_pitch < np.radians(-130))
            or (elbow_pitch > np.radians(130))
        ):
            if verbose:
                print(yellow + str(np.degrees(joints)) + reset_color)
            out_of_limits += 1
        else:
            if verbose:
                print(blue + str(np.degrees(joints)) + reset_color)
        go_to_position(placo_ik, joints, wait=0.0)
        goal_pose_matrix = placo_ik.robot.get_T_a_b("torso", "r_tip_joint")
        # print(goal_pose_matrix)
        position = goal_pose_matrix[:3, 3]
        orientation = R.from_matrix(goal_pose_matrix[:3, :3]).as_euler("xyz")
        goal_pose = [position, orientation]
        is_reachable, joints_placo, errors = placo_ik.is_pose_reachable(
            goal_pose_matrix,
            arm_name="r_arm",
            q0=[0.0, -math.pi / 2, 0.0, -math.pi / 2, 0, 0.0, 0.0],
            tolerances=[0.001, 0.001, 0.001, 0.02, 0.02, 0.02],
            max_iter=45,
            nb_stepper_solve=25,
        )
        result = symbolic_ik.is_reachable(goal_pose)
        if verbose:
            if is_reachable:
                print(green + "Placo reachable" + reset_color)
            else:
                print(red + "Placo not reachable" + reset_color)
            if result[0]:
                print(green + "Symbolic reachable" + reset_color)
                is_reachable, theta = shoulder_limits(result[1], result[2])
                # if is_reachable:
                #     joints, elbow_position = result[2](theta)
                #     go_to_position(placo_ik, joints, wait=0.5)
                # else:
                #     print(red + "Pose not reachable because of shoulder limits" + reset_color)
            else:
                print(red + "Symbolic not reachable" + reset_color)
            time.sleep(0.2)

        if is_reachable:
            placo_success += 1
        if result[0]:
            symbolic_success += 1

    print("JOINTS SPACE TEST")
    print("Number of points : ", number_of_point)
    print("Out of limits : ", out_of_limits)
    print("Symbolic success : ", symbolic_success)
    print("Placo success : ", placo_success)


def task_space_test(
    symbolic_ik: SymbolicIK,
    placo_ik: IKReachyQP,
    x_step: float = 0.15,
    y_step: float = 0.15,
    z_step: float = 0.15,
    roll_step: int = 45,
    pitch_step: int = 45,
    yaw_step: int = 45,
) -> None:
    print("TASK SPACE TEST")
    shoulder_position = symbolic_ik.shoulder_position
    arm_length = symbolic_ik.upper_arm_size + symbolic_ik.forearm_size + symbolic_ik.gripper_size
    arm_length = 0.5
    goal_poses = []
    reachable_poses = 0
    start_time = time.time()
    for x in np.arange(shoulder_position[0] - arm_length, shoulder_position[0] + arm_length + x_step, x_step):
        for y in np.arange(shoulder_position[1] - arm_length, shoulder_position[1] + arm_length + y_step, y_step):
            for z in np.arange(shoulder_position[2] - arm_length, shoulder_position[2] + arm_length + z_step, z_step):
                goal_position = (x, y, z)
                # verify if the position is in the sphere of the arm and is if it's only in front of the robot
                if (np.linalg.norm(np.array(goal_position) - np.array(shoulder_position)) > arm_length) or (
                    goal_position[0] < 0
                ):
                    continue
                for roll in np.arange(0, 360, roll_step):
                    for pitch in np.arange(0, 360, pitch_step):
                        for yaw in np.arange(0, 360, yaw_step):
                            goal_orientation = [np.radians(roll), np.radians(pitch), np.radians(yaw)]
                            goal_pose = [goal_position, goal_orientation]
                            goal_poses.append(goal_pose)
    end_time = time.time()
    print("time : ", end_time - start_time)
    print("goal_poses : ", len(goal_poses))

    start_time = time.time()
    for i in range(len(goal_poses)):
        result = symbolic_ik.is_reachable(goal_poses[i])
        if result[0]:
            reachable_poses += 1
    end_time = time.time()
    print("time : ", end_time - start_time)
    print("reachable poses : ", reachable_poses)
    print("total poses : ", len(goal_poses))


def main_test() -> None:
    symbolib_ik = SymbolicIK()
    urdf_path = Path("src/config_files")
    for file in urdf_path.glob("**/*.urdf"):
        if file.stem == "reachy2_placo":
            urdf_path = file.resolve()
            break
    print(urdf_path)
    placo_ik = IKReachyQP(
        viewer_on=True,
        collision_avoidance=False,
        parts=["r_arm"],
        position_weight=1.9,
        orientation_weight=1e-2,
        robot_version="reachy_2",
        velocity_limit=50.0,
    )
    placo_ik.setup(urdf_path=str(urdf_path))
    placo_ik.create_tasks()

    # time_test(symbolib_ik, placo_ik)
    # joints_space_test(symbolib_ik, placo_ik, verbose=False, number_of_point=1000)
    joints_space_test(symbolib_ik, placo_ik, verbose=True)
    # task_space_test(symbolib_ik, placo_ik)

    # goal_position = [[0.11657383, -0.31879514, -0.18552353], [-2.06584127, -0.50205104, 0.56725307]]
    # result = symbolib_ik.is_reachable(goal_position)
    # print(result[0])

    # joints = [1, -1, -1.5, np.radians(-130), 0, 0, 0]
    # go_to_position(placo_ik, joints, wait=0)
    # goal_pose = placo_ik.robot.get_T_a_b("torso", "r_tip_joint")
    # is_reachable, joints, errors = placo_ik.is_pose_reachable(
    #     goal_pose,
    #     arm_name="r_arm",
    #     q0=[0.0, -math.pi / 2, 0.0, -math.pi / 2, 0, 0.0, 0.0],
    #     tolerances=[0.001, 0.001, 0.001, 0.02, 0.02, 0.02],
    #     max_iter=45,
    #     nb_stepper_solve=25,
    # )
    # print(is_reachable)


if __name__ == "__main__":
    main_test()
