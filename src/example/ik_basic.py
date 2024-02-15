import numpy as np

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK


def main_test() -> None:
    symbolib_ik = SymbolicIK()
    goal_position = [0.0, -0.2, -0.6]
    goal_orientation = [0, 0, 0]
    goal_pose = np.array([goal_position, goal_orientation])
    result = symbolib_ik.is_reachable(goal_pose)
    if result[0]:
        theta = np.linspace(result[1][0], result[1][1], 3)[1]
        joints = result[2](theta)
        print(joints)
    else:
        print("Pose not reachable")


if __name__ == "__main__":
    main_test()
