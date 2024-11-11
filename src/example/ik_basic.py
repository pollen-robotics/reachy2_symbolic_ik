import time

import numpy as np

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import (
    get_best_continuous_theta,
    get_best_continuous_theta2,
    get_best_discrete_theta,
    get_best_theta_to_current_joints,
)


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
    is_reachable_r, interval_r, get_joints_r, _ = symbolic_ik_r.is_reachable(goal_pose)

    previous_theta = np.pi / 2
    preferred_theta = -4 * np.pi / 6
    t = time.time()
    for i in range(1000):
        theta = get_best_continuous_theta(previous_theta, interval_r, get_joints_r, 0.1, preferred_theta, "r_arm")
    print(f" get_best_continuous_theta : {time.time() - t}, state : {theta[2]}")
    t = time.time()
    for i in range(1000):
        theta = get_best_continuous_theta2(previous_theta, interval_r, get_joints_r, 10, 0.1, preferred_theta, "r_arm")
    print(f" get_best_continuous_theta2 : {time.time() - t}, state : {theta[2]}")
    t = time.time()
    for i in range(1000):
        theta = get_best_discrete_theta(previous_theta, interval_r, get_joints_r, 10, preferred_theta, "r_arm")
    print(f" get_best_discrete_theta : {time.time() - t}, state : {theta[2]}")
    t = time.time()
    for i in range(1000):
        theta = get_best_theta_to_current_joints(get_joints_r, 10, [0, 0, 0, 0, 0, 0, 0], "r_arm", preferred_theta)
    print(f" get_best_theta_to_current_joints : {time.time() - t}, state : {theta}")

    if is_reachable_r:
        print("Pose reachable")
        # get joints for one elbow position, define by the angle theta
        theta = interval_r[0]
        t = time.time()
        for i in range(10000):
            joints, elbow_position = get_joints_r(theta)
        print(f" get_joints : {time.time() - t}")
        t = time.time()
        for i in range(10000):
            elbow_position = symbolic_ik_r.get_coordinate_cercle(symbolic_ik_r.intersection_circle, theta)
        print(f" get_coordinate_cercle : {time.time() - t}")

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
    is_reachable_l, interval_l, get_joints_l, _ = symbolic_ik_l.is_reachable(goal_pose)

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
