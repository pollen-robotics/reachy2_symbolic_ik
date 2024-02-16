import numpy as np

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK


def main_test() -> None:
    symbolic_ik_r = SymbolicIK()
    symbolic_ik_l = SymbolicIK(arm="l_arm")

    goal_position = [0.6, -0.2, -0.1]
    goal_orientation = [0, np.radians(-80), 0]
    goal_pose = np.array([goal_position, goal_orientation])
    result_r = symbolic_ik_r.is_reachable(goal_pose)
    if result_r[0]:
        theta = np.linspace(result_r[1][0], result_r[1][1], 3)[1]
        joints = result_r[2](theta)
        print(joints)
    else:
        print("Pose not reachable")

    goal_position = [0.6, 0.2, -0.1]
    goal_orientation = [0, np.radians(-80), 0]
    goal_pose = np.array([goal_position, goal_orientation])
    result_l = symbolic_ik_l.is_reachable(goal_pose)
    if result_l[0]:
        theta = np.linspace(result_l[1][0], result_l[1][1], 3)[1]
        joints = result_l[2](theta)
        print(joints)
    else:
        print("Pose not reachable")


if __name__ == "__main__":
    main_test()
