import numpy as np

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK


def main_test() -> None:
    # Create the symbolic IK for each arm
    symbolic_ik_r = SymbolicIK()
    symbolic_ik_l = SymbolicIK(arm="l_arm")

    # Right hand
    # Define the goal position and orientation
    goal_position = [0.55, -0.3, -0.15]
    goal_orientation = [0, -np.pi / 2, 0]
    goal_pose = np.array([goal_position, goal_orientation])

    # Check if the goal pose is reachable
    is_reachable_r, interval_r, get_joints_r = symbolic_ik_r.is_reachable(goal_pose)

    if is_reachable_r:
        print("Pose reachable")
        # get joints for one elbow position, define by the angle theta
        theta = interval_r[0]
        joints, elbow_position = get_joints_r(theta)
        print(interval_r)
        print(joints)

    else:
        print("Pose not reachable")

    # Left hand
    # Define the goal position and orientation
    goal_position = [0.55, 0.3, -0.15]
    goal_orientation = [0, -np.pi / 2, 0]
    goal_pose = np.array([goal_position, goal_orientation])

    # Check if the goal pose is reachable
    is_reachable_l, interval_l, get_joints_l = symbolic_ik_l.is_reachable(goal_pose)

    if is_reachable_l:
        print("Pose reachable")
        # get joints for one elbow position, define by the angle theta
        theta = interval_l[0]
        joints, elbow_position = get_joints_l(theta)
        print(interval_l)
        print(joints)

    else:
        print("Pose not reachable")


if __name__ == "__main__":
    main_test()
