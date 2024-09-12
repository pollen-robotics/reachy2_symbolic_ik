import copy
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import (
    allow_multiturn,
    get_best_continuous_theta,
    get_best_discrete_theta,
    get_best_theta_to_current_joints,
    get_euler_from_homogeneous_matrix,
    get_ik_parameters_from_urdf,
    limit_orbita3d_joints_wrist,
    limit_theta_to_interval,
    tend_to_preferred_theta,
)

DEBUG = False


class ControlIK:
    def __init__(  # noqa: C901
        # TODO : default current position depends of the shoulder offset
        self,
        current_joints: list[list[float]] = [
            # arms along the body
            [0.0, 0.2617993877991494, -0.17453292519943295, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.2617993877991494, 0.17453292519943295, 0.0, 0.0, 0.0, 0.0],
        ],
        current_pose: list[npt.NDArray[np.float64]] = [
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, -0.2],
                    [0, 0, 1, -0.66],
                    [0, 0, 0, 1],
                ]
            ),
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0.2],
                    [0, 0, 1, -0.66],
                    [0, 0, 0, 1],
                ]
            ),
        ],
        logger: Any = None,
        urdf: str = "",
        urdf_path: str = "",
        reachy_model: str = "full_kit",
    ) -> None:
        self.symbolic_ik_solver = {}
        self.last_call_t = {}
        self.call_timeout = 0.2

        self.nb_search_points = 20

        self.preferred_theta: Dict[str, float] = {}
        self.previous_theta: Dict[str, float] = {}
        self.previous_sol: Dict[str, list[float]] = {}
        self.previous_pose: Dict[str, npt.NDArray[np.float64]] = {}
        self.orbita3D_max_angle = np.deg2rad(42.5)
        self.logger = logger

        if urdf_path == "" and urdf == "":
            raise ValueError("No URDF provided")

        ik_parameters = {}

        if urdf_path != "" and urdf == "":
            urdf_path = os.path.join(os.path.dirname(__file__), urdf_path)
            if os.path.isfile(urdf_path) and os.path.getsize(urdf_path) > 0:
                with open(urdf_path, "r") as fichier:
                    urdf = fichier.read()
            if urdf == "":
                raise ValueError("Empty URDF file")
        if reachy_model == "full_kit" or reachy_model == "headless":
            arms = ["r", "l"]
        elif reachy_model == "starter_kit_right":
            arms = ["r"]
        elif reachy_model == "starter_kit_left":
            arms = ["l"]
        elif reachy_model == "mini":
            arms = []
        else:
            raise ValueError(f"Unknown Reachy model {reachy_model}")

        try:
            ik_parameters = get_ik_parameters_from_urdf(urdf, arms)
        except Exception as e:
            raise ValueError(f"Error while parsing URDF: {e}")

        for prefix in arms:
            arm = f"{prefix}_arm"
            # self.symbolic_ik_solver[arm] = SymbolicIK(
            #     arm=arm,
            #     wrist_limit=np.rad2deg(self.orbita3D_max_angle),
            # )
            if ik_parameters != {}:
                if DEBUG:
                    print(f"Using URDF parameters for {arm}")
                self.symbolic_ik_solver[arm] = SymbolicIK(
                    arm=arm,
                    ik_parameters=ik_parameters,
                )
            else:
                self.symbolic_ik_solver[arm] = SymbolicIK(
                    arm=arm,
                    wrist_limit=np.rad2deg(self.orbita3D_max_angle),
                )

            preferred_theta = -4 * np.pi / 6
            if prefix == "r":
                self.preferred_theta[arm] = preferred_theta
                self.previous_sol[arm] = current_joints[0]
                self.previous_pose[arm] = current_pose[0]
            else:
                self.preferred_theta[arm] = -np.pi - preferred_theta
                self.previous_sol[arm] = current_joints[1]
                self.previous_pose[arm] = current_pose[1]
            current_goal_position, current_goal_orientation = get_euler_from_homogeneous_matrix(self.previous_pose[arm])
            current_pose_tuple = np.array([current_goal_position, current_goal_orientation])
            is_reachable, interval, theta_to_joints_func = self.symbolic_ik_solver[arm].is_reachable_no_limits(
                current_pose_tuple,
            )

            best_prev_theta, state = get_best_theta_to_current_joints(
                theta_to_joints_func,
                20,
                current_joints,
                arm,
                self.preferred_theta[arm],
            )
            self.previous_theta[arm] = best_prev_theta
            self.last_call_t[arm] = 0.0

    def symbolic_inverse_kinematics(  # noqa: C901
        self,
        name: str,
        M: npt.NDArray[np.float64],
        control_type: str,
        current_joints: list[float] = [],
        constrained_mode: str = "unconstrained",
        current_pose: npt.NDArray[np.float64] = np.array([]),
        d_theta_max: float = 0.01,
        preferred_theta: float = -4 * np.pi / 6,
    ) -> Tuple[list[float], bool, str]:
        """
        Compute the inverse kinematics of the goal pose M.
        Args:
            name: r_arm or l_arm
            M: 4x4 homogeneous matrix of the goal pose
            control_type: continuous or discrete
            current_joints: current joints of the arm
            constrained_mode: unconstrained or low_elbow
            current_pose: current pose of the arm
            d_theta_max: maximum angle difference between two consecutive theta
            preferred_theta: preferred theta of the right arm
        Returns:
            ik_joints: list of the joints angles
            is_reachable: True if the goal pose is reachable
            state: if not reachable, the reason why
        """
        goal_position, goal_orientation = get_euler_from_homogeneous_matrix(M)
        goal_pose = np.array([goal_position, goal_orientation])

        if DEBUG:
            print(f"{name} goal_pose: {goal_pose}")
            print(f"{name} control_type: {control_type}")
            print(f"{name} constrained_mode: {constrained_mode}")
            print(f"{name} preferred_theta: {preferred_theta}")

        if constrained_mode == "unconstrained":
            interval_limit = np.array([-np.pi, np.pi])
        elif constrained_mode == "low_elbow":
            interval_limit = np.array([-4 * np.pi / 5, 0])
            # interval_limit = np.array([-4 * np.pi / 5, -np.pi / 2])

        if len(current_pose) == 0:
            current_pose = self.previous_pose[name]

        if current_joints == []:
            current_joints = self.previous_sol[name]

        if name.startswith("l"):
            interval_limit = np.array([-np.pi - interval_limit[1], -np.pi - interval_limit[0]])
            preferred_theta = -np.pi - preferred_theta

        if control_type == "continuous":
            ik_joints, is_reachable, state = self.symbolic_inverse_kinematics_continuous(
                name, goal_pose, interval_limit, current_joints, current_pose, preferred_theta, d_theta_max
            )
        elif control_type == "discrete":
            ik_joints, is_reachable, state = self.symbolic_inverse_kinematics_discrete(
                name, goal_pose, interval_limit, current_joints, preferred_theta
            )
        else:
            raise ValueError(f"Unknown type {control_type}")

        # Test wrist joint limits
        ik_joints_raw = ik_joints
        ik_joints = limit_orbita3d_joints_wrist(ik_joints_raw, self.orbita3D_max_angle)
        if not np.allclose(ik_joints, ik_joints_raw):
            if self.logger is not None:
                self.logger.info(
                    f"{name} Wrist joint limit reached. \nRaw joints: {ik_joints_raw}\nLimited joints: {ik_joints}",
                    throttle_duration_sec=0.1,
                )
            elif DEBUG:
                print(f"{name} Wrist joint limit reached. \nRaw joints: {ik_joints_raw}\nLimited joints: {ik_joints}")

        # Detect multiturns
        ik_joints_allowed = allow_multiturn(ik_joints, self.previous_sol[name], name)
        if not np.allclose(ik_joints_allowed, ik_joints):
            if self.logger is not None:
                self.logger.info(
                    f"{name} Multiturn joint limit reached. \nRaw joints: {ik_joints}\nLimited joints: {ik_joints_allowed}",
                    throttle_duration_sec=0.1,
                )
            elif DEBUG:
                print(f"{name} Multiturn joint limit reached. \nRaw joints: {ik_joints}\nLimited joints: {ik_joints_allowed}")
        ik_joints = ik_joints_allowed
        self.previous_sol[name] = copy.deepcopy(ik_joints)

        # TODO reactivate a smoothing technique

        if DEBUG:
            print(f"{name} ik={ik_joints}")

        self.previous_pose[name] = M

        return ik_joints, is_reachable, state

    def symbolic_inverse_kinematics_continuous(
        self,
        name: str,
        goal_pose: npt.NDArray[np.float64],
        interval_limit: npt.NDArray[np.float64],
        current_joints: list[float],
        current_pose: npt.NDArray[np.float64],
        preferred_theta: float,
        d_theta_max: float,
    ) -> Tuple[list[float], bool, str]:
        """Compute the inverse kinematics of the goal pose M with continuous control.
        Args:
            name: r_arm or l_arm
            goal_pose: position and euler angles of the goal pose
            interval_limit
            current_joints
            current_pose
            preferred_theta
            d_theta_max: maximum angle difference between two consecutive theta
        """

        t = time.time()
        state = ""
        if abs(t - self.last_call_t[name]) > self.call_timeout:
            self.previous_sol[name] = []
            if DEBUG:
                print(f"{name} Timeout reached. Resetting previous_sol {t},  {self.last_call_t[name]}")
        self.last_call_t[name] = t

        if self.previous_sol[name] == []:
            # if the arm moved since last call, we need to update the previous_sol
            # self.previous_sol[name] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # TODO : Get a current position that take the multiturn into consideration
            # Otherwise, when there is no call for more than call_timeout, the joints will be cast between -pi and pi
            # -> If you pause a rosbag during a multiturn and restart it, the previous_sol will be wrong by 2pi
            self.previous_sol[name] = current_joints
            current_goal_position, current_goal_orientation = get_euler_from_homogeneous_matrix(current_pose)
            current_pose_tuple = np.array([current_goal_position, current_goal_orientation])
            is_reachable, interval, theta_to_joints_func = self.symbolic_ik_solver[name].is_reachable_no_limits(
                current_pose_tuple,
            )
            best_prev_theta, state_previous_theta = get_best_theta_to_current_joints(
                theta_to_joints_func, 20, current_joints, name, preferred_theta
            )
            self.previous_theta[name] = best_prev_theta

            if DEBUG:
                print(f"{name}, previous_theta: {self.previous_theta[name]}")

        (
            is_reachable,
            interval,
            theta_to_joints_func,
            state_reachable,
        ) = self.symbolic_ik_solver[
            name
        ].is_reachable(goal_pose)
        # self.print_log(f"{name} state_reachable: {state_reachable}")
        if is_reachable:
            is_reachable, theta, state_theta = get_best_continuous_theta(
                self.previous_theta[name],
                interval,
                theta_to_joints_func,
                d_theta_max,
                preferred_theta,
                self.symbolic_ik_solver[name].arm,
            )
            # is_reachable, theta, state_theta = get_best_continuous_theta2(
            #     self.previous_theta[name],
            #     interval,
            #     theta_to_joints_func,
            #     10,
            #     d_theta_max,
            #     self.preferred_theta[name],
            #     self.symbolic_ik_solver[name].arm,
            # )
            if not is_reachable:
                state = "limited by shoulder"
            theta, state_interval = limit_theta_to_interval(theta, self.previous_theta[name], interval_limit)
            self.previous_theta[name] = theta
            ik_joints, elbow_position = theta_to_joints_func(theta, previous_joints=self.previous_sol[name])

        else:
            if DEBUG:
                print(f"{name} Pose not reachable before even reaching theta selection. State: {state_reachable}")
            is_reachable_no_limits, interval, theta_to_joints_func = self.symbolic_ik_solver[name].is_reachable_no_limits(
                goal_pose
            )
            if is_reachable_no_limits:
                is_reachable_no_limits, theta = tend_to_preferred_theta(
                    self.previous_theta[name],
                    interval,
                    theta_to_joints_func,
                    d_theta_max,
                    goal_theta=preferred_theta,
                )
                theta, state = limit_theta_to_interval(theta, self.previous_theta[name], interval_limit)
                self.previous_theta[name] = theta
                ik_joints, elbow_position = theta_to_joints_func(theta, previous_joints=self.previous_sol[name])
            else:
                print(f"{name} Pose not reachable, this has to be fixed by projecting far poses to reachable sphere")
                raise RuntimeError("Pose not reachable in symbolic IK. We crash on purpose while we are on the debug sessions.")
            state = state_reachable

        if DEBUG:
            print(f"State: {state}")

        return ik_joints, is_reachable, state

    def symbolic_inverse_kinematics_discrete(
        self,
        name: str,
        goal_pose: npt.NDArray[np.float64],
        interval_limit: npt.NDArray[np.float64],
        current_joints: list[float],
        preferred_theta: float,
    ) -> Tuple[list[float], bool, str]:
        """
        Compute the inverse kinematics of the goal pose M with discrete control.
        Args:
            name: r_arm or l_arm
            goal_pose: position and euler angles of the goal pose
            interval_limit
            current_joints
            preferred_theta
        """
        # Checks if an interval exists that handles the wrist limits and the elbow limits
        # self.print_log(f"{name} interval_limit: {interval_limit}")

        (
            is_reachable,
            interval,
            theta_to_joints_func,
            state_reachable,
        ) = self.symbolic_ik_solver[
            name
        ].is_reachable(goal_pose)
        state = state_reachable
        if is_reachable:
            # Explores the interval to find a solution with no collision elbow-torso
            is_reachable, theta, state_theta = get_best_discrete_theta(
                self.previous_theta[name],
                interval,
                theta_to_joints_func,
                self.nb_search_points,
                preferred_theta,
                self.symbolic_ik_solver[name].arm,
            )

            if not is_reachable:
                state = "limited by shoulder"

        if is_reachable:
            theta, state_interval = limit_theta_to_interval(theta, self.previous_theta[name], interval_limit)
            ik_joints, elbow_position = theta_to_joints_func(theta, previous_joints=self.previous_sol[name])
        else:
            ik_joints = current_joints

        return ik_joints, is_reachable, state
