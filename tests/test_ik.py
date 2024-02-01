import numpy as np

# from reachy_placo.ik_reachy_placo import IKReachyQP
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.ik_symbolic import IK_symbolic


def test_class() -> None:
    symbolic_ik = IK_symbolic(upper_arm_size=0.28, forearm_size=0.28, gripper_size=0.15)
    # placo_ik = IKReachyQP()
    assert symbolic_ik is not None

    goal_position = [0.4, 0.2, 0.1]
    goal_orientation = [-60, -90, 20]
    goal_orientation = np.deg2rad(goal_orientation)
    goal_pose = [goal_position, goal_orientation]

    result = symbolic_ik.is_reachable(goal_pose)

    assert not (result[0])
    assert result[1] == []
    assert result[2] is None

    goal_position = [0.6, -0.2, -0.0]
    goal_orientation = [-0, -90, 0]
    goal_orientation = np.deg2rad(goal_orientation)
    goal_pose = [goal_position, goal_orientation]

    result = symbolic_ik.is_reachable(goal_pose)

    assert result[0]
    assert result[1] == [0, np.pi * 2]
    assert result[2] is not None

    joints = result[2](0)

    assert len(joints) == 7

    goal_position = [0.0, -0.2, -0.71]
    goal_orientation = [-0, 0, 0]
    goal_orientation = np.deg2rad(goal_orientation)
    goal_pose = [goal_position, goal_orientation]

    result = symbolic_ik.is_reachable(goal_pose)

    assert result[0]
    assert result[1] == [0, np.pi * 2]
    assert result[2] is not None

    joints = result[2](0)

    assert len(joints) == 7
    assert np.allclose(joints[0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # names = ["r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw", "r_elbow_pitch", "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw"]
    # for i in range(len(names)):
    #     placo_ik.robot.set_joint(names[i], joints[i])
    # # placo_ik._tick_viewer()

    # T_torso_tip = placo_ik.robot.get_T_a_b("torso", "r_tip_joint")
    # position = T_torso_tip[:3, 3]
    # orientation = T_torso_tip[:3, :3]
    # orientation = R.from_matrix(orientation).as_euler("xyz")
    # goal_position = goal_pose[0]
    # goal_orientation = goal_pose[1]

    # assert np.isclose(position, goal_position)
    # for i in range(3):
    #     if abs(position[i] - goal_position[i]) > position_tolerance:
    #         print("Position not correct")
    #         return False
    # for i in range(3):
    #     if abs(orientation[i] - goal_orientation[i]) > orientation_tolerance:
    #         print("Orientation not correct")
    #         return False
    # return True
