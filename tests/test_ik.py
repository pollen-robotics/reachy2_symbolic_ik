from pathlib import Path

import numpy as np

# from scipy.spatial.transform import Rotation as R
import pytest
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK


@pytest.mark.cicd
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
    assert result[1][0] >= -np.pi
    assert result[1][1] <= np.pi
    assert result[2] is not None

    joints, elbow_position = result[2](result[1][0])

    assert len(joints) == 7

    goal_position = [0.0001, -0.2, -0.65]
    goal_orientation = [0.0, 0.0, 0.0]
    goal_pose = [goal_position, goal_orientation]

    result = symbolic_ik.is_reachable(goal_pose)

    assert result[0]
    assert np.all(result[1] == [-np.pi, np.pi])
    assert result[2] is not None

    joints, elbow_position = result[2](0)

    assert len(joints) == 7

    goal_position = [0.0, -0.2, -0.65]
    goal_orientation = [0.0, 0.0, 0.0]
    goal_pose = [goal_position, goal_orientation]

    result = symbolic_ik.is_reachable(goal_pose)

    assert not (result[0])

    goal_position = [0.87, -0.2, -0.0]
    goal_orientation = [0.0, -np.pi / 2, 0.0]
    goal_pose = [goal_position, goal_orientation]

    result = symbolic_ik.is_reachable(goal_pose)

    assert not (result[0])

    goal_position = [0.35, -0.2, -0.28]
    goal_orientation = [0.0, -np.pi / 2, 0.0]
    goal_pose = [goal_position, goal_orientation]

    result = symbolic_ik.is_reachable(goal_pose)

    assert result[0]
