import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.control_ik import ControlIK
from reachy2_symbolic_ik.utils import make_homogenous_matrix_from_rotation_matrix


def control_basics() -> None:
    # Create the control IK for the right arm
    control = ControlIK(urdf_path="../config_files/reachy2.urdf")

    # Define the goal position and orientation
    goal_position = [0.55, -0.3, -0.15]
    goal_orientation = [0, -np.pi / 2, 0]
    goal_pose = np.array([goal_position, goal_orientation])
    goal_pose = make_homogenous_matrix_from_rotation_matrix(goal_position, R.from_euler("xyz", goal_orientation).as_matrix())

    # Get joints for the goal pose
    # The control type can be "discrete" or "continuous"
    # If the control type is "discrete", the control will choose the best elbow position for the goal pose
    # If the control type is "continuous", the control will choose a elbow position that insure continuity in the joints
    control_type = "discrete"
    joints, is_reachable, state = control.symbolic_inverse_kinematics("r_arm", goal_pose, control_type)
    if is_reachable:
        print(f"Pose is reachable \nJoints: {joints}")
    else:
        print("Pose not reachable")


if __name__ == "__main__":
    control_basics()
