from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
import numpy as np
from reachy_placo.ik_reachy_placo import IKReachyQP
import math
import time
from scipy.spatial.transform import Rotation as R
from grasping_utils.utils import get_homogeneous_matrix_msg_from_euler
from importlib.resources import files
from reachy2_symbolic_ik.utils import go_to_position


def are_joints_correct(placo_ik: IKReachyQP, joints: list, goal_pose) -> bool:
    go_to_position(placo_ik, joints, wait=0)
    T_torso_tip = placo_ik.robot.get_T_a_b("torso", "r_tip_joint")
    position = T_torso_tip[:3, 3]
    orientation = T_torso_tip[:3, :3]
    orientation = R.from_matrix(orientation).as_euler("xyz")
    goal_position = goal_pose[0]
    goal_orientation = goal_pose[1]
    print("Position : ", position)
    print("Orientation : ", orientation)
    print("Goal Position : ", goal_position)
    print("Goal Orientation : ", goal_pose[1])
    if not (np.allclose(position, goal_position)):
        print("Position not correct")
        return False
    if not (np.allclose(orientation, goal_orientation)):
        print("Orientation not correct")
        return False
    return True


def main_test() -> None:
    symbolib_ik = SymbolicIK()
    urdf_path = files("config_files")
    for file in urdf_path.glob("**/*.urdf"):
        if file.stem == "reachy2":
            urdf_path = file.resolve()
            break
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

    goal_position = [0.4, 0.1, -0.4]
    goal_orientation = [20, -80, 10]
    goal_orientation = np.deg2rad(goal_orientation)
    goal_pose = [goal_position, goal_orientation]
    result = symbolib_ik.is_reachable(goal_pose)
    if result[0]:
        joints = result[2](result[1][0])
        go_to_position(placo_ik, joints, wait=8)
        joints = result[2](result[1][0])
        is_correct = are_joints_correct(placo_ik, joints, goal_pose)
        print(is_correct)
    else:
        print("Pose not reachable")


if __name__ == "__main__":
    main_test()
