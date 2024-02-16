from pathlib import Path

import numpy as np

# from scipy.spatial.transform import Rotation as R
import pytest
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK


@pytest.mark.noplaco
def test_class() -> None:
    symbolic_ik = SymbolicIK(upper_arm_size=0.28, forearm_size=0.28, gripper_size=0.15)
    assert symbolic_ik is not None

    goal_position = [0.4, 0.2, 0.1]
    # goal_orientation = [-60, -90, 20]
    goal_orientation = [np.radians(-60), np.radians(-90), np.radians(20)]
    goal_pose = [goal_position, goal_orientation]

    result = symbolic_ik.is_reachable(goal_pose)

    assert not (result[0])
    assert len(result[1]) == 0
    assert result[2] is None

    goal_position = [0.3, -0.2, -0.3]
    goal_orientation = [np.radians(0), np.radians(-90), np.radians(0)]
    goal_pose = [goal_position, goal_orientation]

    result = symbolic_ik.is_reachable(goal_pose)

    assert result[0]
    assert result[1][0] >= 0
    assert result[1][1] <= 4 * np.pi
    assert result[2] is not None

    joints = result[2](result[1][0])

    assert len(joints) == 7

    goal_position = [0.0, -0.2, -0.71]
    goal_orientation = [0.0, 0.0, 0.0]
    goal_pose = [goal_position, goal_orientation]

    result = symbolic_ik.is_reachable(goal_pose)

    assert result[0]
    assert np.all(result[1] == [0, np.pi * 2])
    assert result[2] is not None

    joints = result[2](0)

    assert len(joints) == 7


def test_full() -> None:
    from reachy_placo.ik_reachy_placo import IKReachyQP

    symbolic_ik = SymbolicIK(upper_arm_size=0.28, forearm_size=0.28, gripper_size=0.10)
    assert symbolic_ik is not None

    urdf_path = Path("src/config_files")
    for file in urdf_path.glob("**/*.urdf"):
        if file.stem == "reachy2_placo":
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

    goal_position = [0.0, -0.2, -0.65]
    goal_orientation = [0.0, 0.0, 0.0]
    goal_pose = [goal_position, goal_orientation]

    result = symbolic_ik.is_reachable(goal_pose)
    joints = result[2](0)

    names = [
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_elbow_yaw",
        "r_elbow_pitch",
        "r_wrist_roll",
        "r_wrist_pitch",
        "r_wrist_yaw",
    ]
    for i in range(len(names)):
        placo_ik.robot.set_joint(names[i], joints[i])
    placo_ik._tick_viewer()

    T_torso_tip = placo_ik.robot.get_T_a_b("torso", "r_tip_joint")
    position = T_torso_tip[:3, 3]
    orientation = T_torso_tip[:3, :3]
    orientation = R.from_matrix(orientation).as_euler("xyz")
    goal_position = goal_pose[0]
    goal_orientation = goal_pose[1]

    assert np.allclose(position, goal_position)
    assert np.allclose(orientation, goal_orientation)
