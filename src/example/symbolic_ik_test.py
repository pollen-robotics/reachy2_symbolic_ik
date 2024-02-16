from pathlib import Path

import numpy as np
import numpy.typing as npt
from reachy_placo.ik_reachy_placo import IKReachyQP
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils_placo import go_to_position


def are_joints_correct(
    placo_ik: IKReachyQP, joints: npt.NDArray[np.float64], goal_pose: npt.NDArray[np.float64], arm: str = "r_arm"
) -> bool:
    go_to_position(placo_ik, joints, wait=0)
    if arm == "r_arm":
        T_torso_tip = placo_ik.robot.get_T_a_b("torso", "r_tip_joint")
    else:
        T_torso_tip = placo_ik.robot.get_T_a_b("torso", "l_tip_joint")
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
        print("Position : ", position)
        print("Goal Position : ", goal_position)
        return False
    if not (np.allclose(orientation, goal_orientation)):
        print("Orientation not correct")
        return False
    return True


def main_test() -> None:
    symbolib_ik_r = SymbolicIK()
    symbolib_ik_l = SymbolicIK(arm="l_arm")
    urdf_path = Path("src/config_files")
    for file in urdf_path.glob("**/*.urdf"):
        if file.stem == "reachy2_ik":
            urdf_path = file.resolve()
            break
    placo_ik = IKReachyQP(
        viewer_on=True,
        collision_avoidance=False,
        parts=["r_arm", "l_arm"],
        position_weight=1.9,
        orientation_weight=1e-2,
        robot_version="reachy_2",
        velocity_limit=50.0,
    )

    placo_ik.setup(urdf_path=str(urdf_path))
    placo_ik.create_tasks()
    # joints = [0, -0, -0, 0, 0, 0, 0]

    # go_to_position(placo_ik, [0.5, -np.radians(0), 0, np.radians(-0), 0, 0, 0], arm="l_arm", wait=0)
    # go_to_position(placo_ik, [0.5, -np.radians(0), 0, np.radians(-0), 0, 0, 0], arm="r_arm", wait=5)
    # print(placo_ik.robot.get_T_a_b("torso", "r_tip_joint"))
    # print(placo_ik.robot.get_T_a_b("torso", "r_wrist_roll"))
    # print(placo_ik.robot.get_T_a_b("torso", "r_elbow_yaw"))
    # go_to_position(placo_ik, [0, -np.radians(-0), 0, np.radians(337), 0, 0, 0], arm="l_arm", wait=5)

    goal_position = [0.60, 0.2, -0.1]
    goal_orientation = [0, -np.radians(70), 0]
    goal_pose = np.array([goal_position, goal_orientation])

    result_l = symbolib_ik_l.is_reachable(goal_pose)
    if result_l[0]:
        print(result_l[1])
        theta = np.linspace(result_l[1][0], result_l[1][1], 3)[1]
        joints = result_l[2](theta)
        go_to_position(placo_ik, joints, wait=3, arm="l_arm")
        print(np.degrees(joints))
        is_correct = are_joints_correct(placo_ik, joints, goal_pose, arm="l_arm")
        print(is_correct)
    else:
        print("Pose not reachable")

    # goal_position = [0.60, -0.20, -0.1]
    # goal_orientation = [0, -np.radians(70), 0]
    # goal_pose = np.array([goal_position, goal_orientation])
    goal_pose = [
        [0.10000000000000009, -0.5499999999999999, 0.2500000000000001],
        [4.71238898038469, 5.497787143782138, 1.5707963267948966],
    ]

    result_r = symbolib_ik_r.is_reachable(goal_pose)
    if result_r[0]:
        print("pose reachable")
        print(result_r[1])
        theta = np.linspace(result_r[1][0], result_r[1][1], 3)[1]
        joints = result_r[2](theta)
        go_to_position(placo_ik, joints, wait=3)
        is_correct = are_joints_correct(placo_ik, joints, goal_pose)
        print(is_correct)

    else:
        print("Pose not reachable")


if __name__ == "__main__":
    main_test()
