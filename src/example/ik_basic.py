import numpy as np
import time

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK


def main_test() -> None:
    symbolic_ik_r = SymbolicIK()
    symbolic_ik_l = SymbolicIK(arm="l_arm")



    # goal_position = [-0.5, -0.2, 0.1]
    # goal_orientation = [0, np.pi/2,0]
    # goal_pose = np.array([goal_position, goal_orientation])
    # result_r = symbolic_ik_r.is_reachable(goal_pose)

    # if result_r[0]:
    #     joints, elbow_position = result_r[2](result_r[1][0])
    #     print(result_r[1])
    #     print(joints)

    # else:
    #     print("Pose not reachable")

    # goal_position = [-0.5, -0.2, 0.]
    # goal_orientation = [0, np.pi/2,0]
    # goal_pose = np.array([goal_position, goal_orientation])
    # result_r = symbolic_ik_r.is_reachable(goal_pose)

    # if result_r[0]:
    #     joints, elbow_position = result_r[2](result_r[1][0])
    #     print(result_r[1])
    #     print(joints)

    # else:
    #     print("Pose not reachable")

    # goal_position = [-0.5, -0.2, -0.2]
    # goal_orientation = [0, np.pi/2,0]
    # goal_pose = np.array([goal_position, goal_orientation])
    # result_r = symbolic_ik_r.is_reachable(goal_pose)

    # if result_r[0]:
    #     joints, elbow_position = result_r[2](result_r[1][0])
    #     print(result_r[1])
    #     print(joints)

    # else:
    #     print("Pose not reachable")

    # time.sleep(2)

    goal_position = [-0.55, -0.2, -0.0]
    goal_orientation = [0, np.pi/2,0]
    goal_pose = np.array([goal_position, goal_orientation])
    result_r = symbolic_ik_r.is_reachable(goal_pose)

    if result_r[0]:
        joints, elbow_position = result_r[2](result_r[1][0])
        print(result_r[1])
        print(joints)

    else:
        print("Pose not reachable")

    goal_position = [0.55, -0.3, -0.0]
    goal_orientation = [0, -np.pi/2,0]
    goal_pose = np.array([goal_position, goal_orientation])
    result_r = symbolic_ik_r.is_reachable(goal_pose)

    if result_r[0]:
        joints, elbow_position = result_r[2](result_r[1][0])
        print(result_r[1])
        print(joints)

    else:
        print("Pose not reachable")


    goal_position = [-0.0, -0.75, -0.0]
    goal_orientation = [-np.pi/2, 0,0]
    goal_pose = np.array([goal_position, goal_orientation])
    result_r = symbolic_ik_r.is_reachable(goal_pose)

    if result_r[0]:
        joints, elbow_position = result_r[2](result_r[1][0])
        print(result_r[1])
        print(joints)

    else:
        print("Pose not reachable")



    



    # goal_position = [-0.4, -0.4, -0.3]
    # goal_orientation = [0, np.pi/2,0]
    # goal_pose = np.array([goal_position, goal_orientation])
    # result_r = symbolic_ik_r.is_reachable(goal_pose)

    # if result_r[0]:
    #     joints, elbow_position = result_r[2](result_r[1][0])
    #     print(result_r[1])
    #     print(joints)

    # else:
    #     print("Pose not reachable")

    # goal_position = [-0.3, -0.4, -0.4]
    # goal_orientation = [0, 0, 0]
    # goal_pose = np.array([goal_position, goal_orientation])
    # result_r = symbolic_ik_r.is_reachable(goal_pose)

    # if result_r[0]:
    #     joints, elbow_position = result_r[2](result_r[1][0])
    #     print(result_r[1])
    #     print(joints)

    # else:
    #     print("Pose not reachable")

    # # time.sleep(2)

    # goal_position = [0.3, -0.4, -0.4]
    # goal_orientation = [0, 0,0]
    # goal_pose = np.array([goal_position, goal_orientation])
    # result_r = symbolic_ik_r.is_reachable(goal_pose)

    # if result_r[0]:
    #     joints, elbow_position = result_r[2](result_r[1][0])
    #     print(result_r[1])
    #     print(joints)

    # else:
    #     print("Pose not reachable")

    # goal_position = [0.55, 0.2, -0.1]
    # goal_orientation = [0, np.radians(-80), 0]
    # goal_pose = np.array([goal_position, goal_orientation])
    # result_l = symbolic_ik_l.is_reachable(goal_pose)
    # if result_l[0]:
    #     is_reachable, theta = shoulder_limits(result_l[1], result_l[2], arm="l_arm")
    #     if is_reachable:
    #         joints, elbow_position = result_l[2](theta)
    #         print(joints)
    #     else:
    #         print("Pose not reachable because of shoulder limits")
    # else:
    #     print("Pose not reachable")


if __name__ == "__main__":
    main_test()
