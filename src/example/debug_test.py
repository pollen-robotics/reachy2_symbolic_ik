import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import get_best_continuous_theta, get_valid_arm_joints, tend_to_prefered_theta


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

    num_samples = 1

    for i in range(num_samples):
        # Sample random poses within the specified ranges
        x, y, z = (
            np.random.uniform(x_range[0], x_range[1]),
            np.random.uniform(y_range[0], y_range[1]),
            np.random.uniform(z_range[0], z_range[1]),
        )
        roll, pitch, yaw = np.random.uniform(angle_range[0], angle_range[1], 3)

        goal_pose = np.array([[x, y, z], [roll, pitch, yaw]])

        # Compare results between r_arm and l_arm
        is_reachable_r, interval_r, get_joints_r = symbolic_ik_solver["r_arm"].is_reachable_no_limits(goal_pose)
        is_reachable_l, interval_l, get_joints_l = symbolic_ik_solver["l_arm"].is_reachable_no_limits(goal_pose)

        print(f"Sample {i+1}:")
        print(f"Goal Pose: {np.round(goal_pose[0], 2).tolist()}, {np.round(goal_pose[1], 2).tolist()}")

        for theta in [0]:  # [-np.pi / 4, 0, np.pi / 2]:
            r_joints, _ = get_joints_r(theta)
            l_joints, _ = get_joints_l(theta)
            print(f"r_joints: {np.round(r_joints, 2).tolist()}")
            print(f"l_joints: {np.round(l_joints, 2).tolist()}")


if __name__ == "__main__":
    main_test()
