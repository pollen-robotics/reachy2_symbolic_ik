import numpy as np

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK


def ik_basics() -> None:
    # Create the symbolic IK for the right arm
    symbolic_ik = SymbolicIK(arm="r_arm")

    # Define the goal position and orientation
    goal_position = [0.55, -0.3, -0.15]
    goal_orientation = [0, -np.pi / 2, 0]
    goal_pose = np.array([goal_position, goal_orientation])

    # Check if the goal pose is reachable
    is_reachable, theta_interval, theta_to_joints_func, state = symbolic_ik.is_reachable(goal_pose)

    # Get the joints for one elbow position, defined by the angle theta
    if is_reachable:
        # Choose a theta in the interval
        # if theta_interval[0] < theta_interval[1], theta can be any value in the interval
        # else theta can be in the intervals [-np.pi, theta_interval[1]] or [theta_interval[0], np.pi]
        theta = theta_interval[0]

        # Get the joints
        joints, elbow_position = theta_to_joints_func(theta)
        print(f"Pose is reachable \nJoints: {joints}")
    else:
        print("Pose not reachable")


if __name__ == "__main__":
    ik_basics()
