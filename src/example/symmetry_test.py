import time

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK


def get_euler_from_homogeneous_matrix(
    homogeneous_matrix: npt.NDArray[np.float64], degrees: bool = False
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    position = homogeneous_matrix[:3, 3]
    rotation_matrix = homogeneous_matrix[:3, :3]
    euler_angles = np.array(Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=degrees))
    return position, euler_angles


def build_pose_matrix(x: np.float64, y: np.float64, z: np.float64) -> npt.NDArray[np.float64]:
    # The effector is always at the same orientation in the world frame
    return np.array(
        [
            [0.0, 0.0, -1.0, x],
            [0.0, 1.0, 0.0, y],
            [1.0, 0.0, 0.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def main_test() -> None:
    symbolic_ik_solver = {}

    for prefix in ("l", "r"):
        arm = f"{prefix}_arm"
        symbolic_ik_solver[arm] = SymbolicIK(
            arm=arm,
            upper_arm_size=0.28,
            forearm_size=0.28,
            gripper_size=0.10,
            wrist_limit=45,
            # This is the "correct" stuff for alpha
            # shoulder_orientation_offset=[10, 0, 15],
            # This is the "wrong" values currently used by the alpha
            shoulder_orientation_offset=[0, 0, 15],
            shoulder_position=[-0.0479, -0.1913, 0.025],
        )

    x_range = (-0.1, 1.0)
    y_range = (-0.50, 0.90)
    z_range = (-0.7, 0.7)
    angle_range = (-np.pi, np.pi)

    r_prefered_theta = 5 * np.pi / 4
    l_prefered_theta = np.pi - r_prefered_theta

    num_samples = 1000
    t0 = time.time()
    for i in range(num_samples):
        # Sample random poses within the specified ranges
        x, y, z = (
            np.random.uniform(x_range[0], x_range[1]),
            np.random.uniform(y_range[0], y_range[1]),
            np.random.uniform(z_range[0], z_range[1]),
        )
        roll, pitch, yaw = np.random.uniform(angle_range[0], angle_range[1], 3)

        # Use this for random angles + random positions
        goal_pose = np.array([[x, y, z], [roll, pitch, yaw]])
        goal_pose_l = np.array([[x, -y, z], [-roll, pitch, -yaw]])

        # Use this for "straight" angles + random positions
        # goal_pose = np.array([[x, y, z], [0, -np.pi / 2, 0]])
        # goal_pose_l = np.array([[x, -y, z], [0, -np.pi / 2, 0]])

        # Compare results between r_arm and l_arm
        is_reachable_r, interval_r, get_joints_r = symbolic_ik_solver["r_arm"].is_reachable_no_limits(goal_pose)
        is_reachable_l, interval_l, get_joints_l = symbolic_ik_solver["l_arm"].is_reachable_no_limits(goal_pose_l)
        not_ok = False
        for theta in [0, -np.pi / 4, np.pi / 2, 3 * np.pi, np.pi]:
            r_joints, _ = get_joints_r(r_prefered_theta + theta)
            l_joints, _ = get_joints_l(l_prefered_theta - theta)
            l_mod = np.array([l_joints[0], -l_joints[1], -l_joints[2], l_joints[3], -l_joints[4], l_joints[5], -l_joints[6]])
            # calculate l2 distance between r_joints and l_mod
            l2_dist = np.linalg.norm(r_joints - l_mod)
            if l2_dist < 0.001:
                # print("OK")
                pass
            else:
                not_ok = True
                print(f"Sample {i+1}:")
                print(f"Goal Pose: {np.round(goal_pose[0], 2).tolist()}, {np.round(goal_pose[1], 2).tolist()}")
                print("Not OK!!")
                print(f"l2_dist: {l2_dist:.2f}")
                print(f"r_joints: {np.round(r_joints, 2).tolist()}")
                print(f"l_mod: {np.round(l_mod, 2).tolist()}")
                # print(f"l_joints: {np.round(l_joints, 2).tolist()}")
        if not_ok:
            break
    t1 = time.time()
    print(f"Elapsed time: {t1-t0:.2f} seconds")
    print(f"All {num_samples} samples gave symmetric results!")


if __name__ == "__main__":
    main_test()
