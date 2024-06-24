import copy
import time
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import (  # get_best_continuous_theta2,
    allow_multiturn,
    # distance_from_singularity,
    get_best_continuous_theta,
    get_best_discrete_theta,
    get_best_theta_to_current_joints,
    get_euler_from_homogeneous_matrix,
    limit_orbita3d_joints_wrist,
    limit_theta_to_interval,
    tend_to_prefered_theta,
)

SHOW_GRAPH = False


class ControlIK:
    def __init__(
        # TODO : default current position depends of the shoulder offset
        self,
        current_joints: list[list[float]] = [
            [0.0, -0.17453292519943295, -0.2617993877991494, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.17453292519943295, 0.2617993877991494, 0.0, 0.0, 0.0, 0.0],
            # [6.44728404e-01, -2.95532788e-01,  1.06260987e-01, -1.27504976e+00,
            #   1.38270728e-10,  6.37524882e-01, -5.23598776e-01],
            # [6.44728404e-01,  2.95532788e-01, -1.06260987e-01, -1.27504976e+00,
            #  -1.38271616e-10,  6.37524882e-01,  5.23598776e-01]
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
    ) -> None:
        self.symbolic_ik_solver = {}
        self.last_call_t = {}
        self.call_timeout = 0.2

        self.nb_search_points = 20

        self.prefered_theta: Dict[str, float] = {}
        self.previous_theta: Dict[str, float] = {}
        self.previous_sol: Dict[str, list[float]] = {}
        self.previous_pose: Dict[str, npt.NDArray[np.float64]] = {}
        self.orbita3D_max_angle = np.deg2rad(42.5)

        self.logger = logger

        for prefix in ("r", "l"):
            arm = f"{prefix}_arm"
            self.symbolic_ik_solver[arm] = SymbolicIK(
                arm=arm,
                wrist_limit=np.rad2deg(self.orbita3D_max_angle),
            )
            if prefix == "r":
                self.prefered_theta[arm] = -4 * np.pi / 6
                self.previous_sol[arm] = current_joints[0]
                self.previous_pose[arm] = current_pose[0]
            else:
                self.prefered_theta[arm] = -np.pi - self.prefered_theta["r_arm"]
                self.previous_sol[arm] = current_joints[1]
                self.previous_pose[arm] = current_pose[1]
            self.previous_theta[arm] = self.prefered_theta[arm]
            self.last_call_t[arm] = 0.0
            self.print_log("laaaa")

    def symbolic_inverse_kinematics(  # noqa: C901
        self,
        name: str,
        M: npt.NDArray[np.float64],
        control_type: str,
        current_joints: list[float] = [],
        interval_limit: npt.NDArray[np.float64] = np.array([]),
        current_pose: npt.NDArray[np.float64] = np.array([]),
    ) -> Tuple[list[float], bool, str]:
        goal_position, goal_orientation = get_euler_from_homogeneous_matrix(M)
        goal_pose = np.array([goal_position, goal_orientation])
        # self.print_log(f"{name} goal_position: {goal_position}")
        # self.print_log(f" controle_type: {control_type}")

        if len(current_pose) == 0:
            current_pose = self.previous_pose[name]
            # if name.startswith("r"):

            #     # current_pose = np.array(
            #     #     [
            #     #         [1, 0, 0, 0],
            #     #         [0, 1, 0, -0.2],
            #     #         [0, 0, 1, -0.66],
            #     #         [0, 0, 0, 1],
            #     #     ]
            #     # )
            # else:
            #     current_pose = np.array(
            #         [
            #             [1, 0, 0, 0],
            #             [0, 1, 0, 0.2],
            #             [0, 0, 1, -0.66],
            #             [0, 0, 0, 1],
            #         ]
            #     )

        if current_joints == []:
            current_joints = self.previous_sol[name]

        if control_type == "continuous":
            if len(interval_limit) == 0:
                interval_limit = np.array([-4 * np.pi / 5, 0])
            if name.startswith("l"):
                interval_limit = np.array([-np.pi - interval_limit[1], -np.pi - interval_limit[0]])
            ik_joints, is_reachable, state = self.symbolic_inverse_kinematics_continuous(
                name, goal_pose, interval_limit, current_joints, current_pose
            )
        elif control_type == "discrete":
            if len(interval_limit) == 0:
                # if interval_limit.size == 0:
                self.print_log(f"{name} No interval limit provided. Using default [-pi, pi]")
                interval_limit = np.array([-np.pi, np.pi])
            if name.startswith("l"):
                self.print_log(f"{name} interval_limit: {interval_limit}")
                interval_limit = np.array([-np.pi - interval_limit[1], -np.pi - interval_limit[0]])
            ik_joints, is_reachable, state = self.symbolic_inverse_kinematics_discrete(
                name, goal_pose, interval_limit, current_joints
            )
        else:
            raise ValueError(f"Unknown type {control_type}")

        # self.print_log(f"{name} ik_brut={ik_joints}")

        ik_joints_raw = ik_joints
        ik_joints = limit_orbita3d_joints_wrist(ik_joints_raw, self.orbita3D_max_angle)
        if not np.allclose(ik_joints, ik_joints_raw):
            self.print_log(f"{name} Wrist joint limit reached. \nRaw joints: {ik_joints_raw}\nLimited joints: {ik_joints}")

        ik_joints_allowed = allow_multiturn(ik_joints, self.previous_sol[name], name)
        if not np.allclose(ik_joints_allowed, ik_joints):
            self.print_log(
                f"{name} Multiturn joint limit reached. \nRaw joints: {ik_joints}\nLimited joints: {ik_joints_allowed}"
            )
        ik_joints = ik_joints_allowed
        # self.logger.info(f"{name} ik={ik_joints}")
        self.previous_sol[name] = copy.deepcopy(ik_joints)
        # self.previous_sol[name] = ik_joints
        # self.logger.info(f"{name} ik={ik_joints}, elbow={elbow_position}")

        # TODO reactivate a smoothing technique
        # self.logger.warning(f"{name} ik={ik_joints}", throttle_duration_sec=0.1)
        # self.print_log(f"{name} ik={ik_joints}")

        self.previous_pose[name] = M

        return ik_joints, is_reachable, state

    def symbolic_inverse_kinematics_continuous(
        self,
        name: str,
        goal_pose: npt.NDArray[np.float64],
        interval_limit: npt.NDArray[np.float64],
        current_joints: list[float],
        current_pose: npt.NDArray[np.float64],
    ) -> Tuple[list[float], bool, str]:
        # self.print_log("continuous")
        t = time.time()
        print(f" last_call_t: {self.last_call_t}")
        print(f" last_call_t: {self.last_call_t[name]}")
        state = ""
        if abs(t - self.last_call_t[name]) > self.call_timeout:
            # self.logger.warning(
            #     f"{name} Timeout reached. Resetting previous_theta and previous_sol"
            # )
            self.previous_sol[name] = []
            self.print_log(f"{name} Timeout reached. Resetting previous_sol {t},  {self.last_call_t[name]}")
        self.last_call_t[name] = t
        print(f" last_call_t: {self.last_call_t[name]}")
        print(f" last_call_t: {self.last_call_t}")
        d_theta_max = 0.01

        # if self.previous_theta[name] is None:
        #     self.previous_theta[name] = self.prefered_theta[name]

        if self.previous_sol[name] == []:
            # if the arm moved since last call, we need to update the previous_sol
            # self.previous_sol[name] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # TODO : Get a current position that take the multiturn into consideration
            # Otherwise, when there is no call for more than call_timeout, the joints will be cast between -pi and pi
            # -> If you pause a rosbag during a multiturn and restart it, the previous_sol will be wrong by 2pi
            self.previous_sol[name] = current_joints

            # Finding the current pose and estimating the best theta to match that pose
            # error, current_pose = forward_kinematics(
            #     self.fk_solver[name],
            #     current_joints,
            #     self.chain[name].getNrOfJoints(),
            # )

            current_goal_position, current_goal_orientation = get_euler_from_homogeneous_matrix(current_pose)
            # self.print_log(f"current_goal_position: {current_goal_position}")
            # self.print_log(f"current_goal_orientation: {current_goal_orientation}")
            # current_pose_tuple = np.array(goal_pose)
            current_pose_tuple = np.array([current_goal_position, current_goal_orientation])
            # t0 = time.time()
            is_reachable, interval, theta_to_joints_func = self.symbolic_ik_solver[name].is_reachable_no_limits(
                current_pose_tuple,
            )
            # self.print_log(f"current_pose: {current_pose_tuple}, arm: {name}")

            # self.print_log(f"current_joints: {current_joints}, arm: {name}")
            best_prev_theta, state = get_best_theta_to_current_joints(
                theta_to_joints_func,
                20,
                current_joints,
                name,
            )
            # t1 = time.time()
            self.previous_theta[name] = best_prev_theta
            # self.print_log(f"{name} previous_sol is None. Setting it to current position : {self.previous_sol[name]}")
            # self.print_log(
            #     f"{name} estimating previous_theta from current position. Best previous theta: {self.previous_theta[name]}"
            # )
            # self.print_log(f"state: {state}. Computation time: {(t1-t0)*1000:.2f}ms")

        # self.logger.warning(
        #     f"{name} prefered_theta: {prefered_theta}, previous_theta: {self.previous_theta[name]}"
        # )
        (
            is_reachable,
            interval,
            theta_to_joints_func,
            state_reachable,
        ) = self.symbolic_ik_solver[
            name
        ].is_reachable(goal_pose)
        if is_reachable:
            is_reachable, theta, state_theta = get_best_continuous_theta(
                self.previous_theta[name],
                interval,
                theta_to_joints_func,
                # 10,
                d_theta_max,
                self.prefered_theta[name],
                self.symbolic_ik_solver[name].arm,
            )
            # is_reachable, theta, state_theta = get_best_continuous_theta2(
            #     self.previous_theta[name],
            #     interval,
            #     theta_to_joints_func,
            #     10,
            #     d_theta_max,
            #     self.prefered_theta[name],
            #     self.symbolic_ik_solver[name].arm,
            # )
            if not is_reachable:
                self.print_log(f"{name} Pose not reachable. State: {state_theta}")
                state = "limited by shoulder"
            # self.print_log(f"name: {name}, theta: {theta}")
            theta, state_interval = limit_theta_to_interval(theta, self.previous_theta[name], interval_limit)
            # self.print_log(f"{name} State interval: {state_interval}")
            # self.print_log(
            #    f"name: {name}, theta: {theta}, previous_theta: {self.previous_theta[name]}, state: {state_theta}"
            # )
            self.previous_theta[name] = theta
            ik_joints, elbow_position = theta_to_joints_func(theta, previous_joints=self.previous_sol[name])
            # self.print_log(
            #    f"{name} Is reachable. Is truly reachable: {is_reachable}. State: {state_theta}"
            # )

        else:
            self.print_log(f"{name} Pose not reachable before even reaching theta selection. State: {state_reachable}")
            is_reachable, interval, theta_to_joints_func = self.symbolic_ik_solver[name].is_reachable_no_limits(goal_pose)
            if is_reachable:
                is_reachable, theta = tend_to_prefered_theta(
                    self.previous_theta[name],
                    interval,
                    theta_to_joints_func,
                    d_theta_max,
                    goal_theta=self.prefered_theta[name],
                )
                theta, state = limit_theta_to_interval(theta, self.previous_theta[name], interval_limit)
                # self.print_log(
                #    f"name: {name}, theta: {theta}, previous_theta: {self.previous_theta[name]}"
                # )
                self.previous_theta[name] = theta
                ik_joints, elbow_position = theta_to_joints_func(theta, previous_joints=self.previous_sol[name])
            else:
                self.print_log(f"{name} Pose not reachable, this has to be fixed by projecting far poses to reachable sphere")
                raise RuntimeError("Pose not reachable in symbolic IK. We crash on purpose while we are on the debug sessions.")
            state = state_reachable

        # self.print_log(f"State: {state}")
        # self.print_log(f"state : {state_theta}")

        return ik_joints, is_reachable, state

    def symbolic_inverse_kinematics_discrete(
        self,
        name: str,
        goal_pose: npt.NDArray[np.float64],
        interval_limit: npt.NDArray[np.float64],
        current_joints: list[float],
    ) -> Tuple[list[float], bool, str]:
        # self.print_log("discrete")
        # Checks if an interval exists that handles the wrist limits and the elbow limits
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
                self.prefered_theta[name],
                self.symbolic_ik_solver[name].arm,
            )

            if not is_reachable:
                self.print_log(f"{name} Pose not reachable. State: {state_theta}")
                state = "limited by shoulder"
            # is_reachable, theta, state = get_best_discrete_theta_min_mouvement(
            #     self.previous_theta[name],
            #     interval,
            #     theta_to_joints_func,
            #     self.nb_search_points,
            #     prefered_theta,
            #     self.symbolic_ik_solver[name].arm,
            #     np.array(self.get_current_joints(self.chain[name]))
            # )
            # self.logger.info(f"state get_best_discrete_theta: {state}")
            # self.logger.info(f"Best theta: {theta}")
        # else:
        #     self.logger.error(f"{name} Pose not reachable before even reaching theta selection. State: {state_reachable}")
        # self.print_log(f"State: {is_reachable}")

        if is_reachable:
            ik_joints, elbow_position = theta_to_joints_func(theta, previous_joints=self.previous_sol[name])
            # singularity_distance = distance_from_singularity(elbow_position[:3], arm=name)
            # self.print_log(f"{name} elbow_position: {elbow_position}")
            # self.print_log(f"{name} Singularity distance: {singularity_distance}")
        else:
            ik_joints = current_joints

        return ik_joints, is_reachable, state

    def print_log(self, msg: str, throttle_duration: float = 0) -> None:
        if self.logger is not None:
            self.logger.info(msg, throttle_duration_sec=throttle_duration)
        else:
            print(msg)
