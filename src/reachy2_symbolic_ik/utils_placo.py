import math
import time
from typing import List

import numpy as np
import numpy.typing as npt
from reachy_placo.ik_reachy_placo import IKReachyQP


def go_to_position(
    reachy_placo: IKReachyQP,
    joint_pose: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0, -math.pi / 2, 0.0, 0.0, 0.0]),
    wait: int = 10,
) -> None:
    """
    Show pose with the r_arm in meshcat
    args:
        joint_pose: joint pose of the arm
        wait: time to wait before closing the window
    """
    names = r_arm_joint_names()
    for i in range(len(names)):
        reachy_placo.robot.set_joint(names[i], joint_pose[i])
    reachy_placo._tick_viewer()
    time.sleep(wait)


def r_arm_joint_names() -> List[str]:
    names = []
    names.append("r_shoulder_pitch")
    names.append("r_shoulder_roll")
    names.append("r_elbow_yaw")
    names.append("r_elbow_pitch")
    names.append("r_wrist_roll")
    names.append("r_wrist_pitch")
    names.append("r_wrist_yaw")
    return names
