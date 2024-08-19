import time

import numpy as np
import numpy.typing as npt
import rclpy
import tf_transformations
from geometry_msgs.msg import Vector3
from rclpy.qos import QoSProfile
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import (  # get_best_continuous_theta2,; distance_from_singularity,
    get_best_discrete_theta,
    get_euler_from_homogeneous_matrix,
    limit_orbita3d_joints_wrist,
)


def add_sphere(markers: MarkerArray, size: float, color: ColorRGBA, position: npt.NDArray[np.float64], index: int = 0) -> None:
    marker = Marker()
    marker.header.frame_id = "torso"
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.scale.x = size
    marker.scale.y = size
    marker.scale.z = size
    marker.color.a = color.a
    marker.color.r = color.r
    marker.color.g = color.g
    marker.color.b = color.b
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.id = index

    markers.markers.append(marker)


def add_frame(markers: MarkerArray, pose: npt.NDArray[np.float64], id: int) -> None:
    colors = [
        ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
        ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
        ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
    ]
    position = pose[0]
    orientation = pose[1]
    roll_axe = R.from_euler("xyz", orientation)
    pitch_axe = roll_axe * R.from_euler("xyz", [0.0, 0.0, np.pi / 2])
    yaw_axe = roll_axe * R.from_euler("xyz", [0.0, np.pi / 2, 0.0])
    roll_axe = roll_axe.as_euler("xyz")
    pitch_axe = pitch_axe.as_euler("xyz")
    yaw_axe = yaw_axe.as_euler("xyz")
    markers.markers.append(create_arrow(position, roll_axe, colors[0], 0.1, id))
    markers.markers.append(create_arrow(position, -pitch_axe, colors[1], 0.1, id + 1))
    markers.markers.append(create_arrow(position, yaw_axe, colors[2], 0.1, id + 2))


def create_arrow(xyz: npt.NDArray[np.float64], rpy: npt.NDArray[np.float64], color: ColorRGBA, size: float, id: int) -> Marker:
    msg = Marker()
    msg.id = id
    msg.frame_locked = True
    msg.action = Marker.ADD
    msg.header.frame_id = "torso"
    msg.type = Marker.ARROW
    msg.mesh_use_embedded_materials = False
    msg.scale = Vector3(x=size, y=0.01, z=0.01)
    msg.color = color
    msg.ns = "Position"
    msg.lifetime.sec = 0
    msg.lifetime.nanosec = 0
    msg.pose.position.x = xyz[0]
    msg.pose.position.y = xyz[1]
    msg.pose.position.z = xyz[2]
    q = tf_transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
    msg.pose.orientation.x = q[0]
    msg.pose.orientation.y = q[1]
    msg.pose.orientation.z = q[2]
    msg.pose.orientation.w = q[3]
    return msg


def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str, markers: MarkerArray, marker_id: int) -> None:
    symbolic_ik = SymbolicIK(arm)
    symbolic_ik.tip_position = np.array([0.05, -0.05, 0.17])

    preferred_theta = -4 * np.pi / 6
    if arm == "l_arm":
        preferred_theta = -np.pi - preferred_theta

    is_reachable, interval, get_joints, state = symbolic_ik.is_reachable(pose)
    if not is_reachable:
        print("Not reachable")
        print(state)
        return
    is_reachable, theta, state = get_best_discrete_theta(preferred_theta, interval, get_joints, 20, preferred_theta, arm)
    if not is_reachable:
        print("Not reachable - no theta found")
        return

    t1 = time.time()
    for i in range(1):
        joints, elbow_position = get_joints(theta)
    t2 = time.time()
    print(f"Time to get joints: {t2 - t1}")

    raw_joints = joints.copy()
    joints = limit_orbita3d_joints_wrist(raw_joints, np.radians(42.5))
    if not np.isclose(joints, raw_joints).all():
        print("NOT REACHABLE")
    joints = np.degrees(joints)

    if arm == "r_arm":
        reachy.r_arm.goto_joints(joints, 3.0, degrees=True, interpolation_mode="minimum_jerk")
    elif arm == "l_arm":
        reachy.l_arm.goto_joints(joints, 3.0, degrees=True, interpolation_mode="minimum_jerk")

    time.sleep(1.0)

    add_sphere(markers, 0.01, ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), pose[0], marker_id)
    add_sphere(markers, 0.09, ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.4), elbow_position, marker_id + 1)
    # add_sphere(markers, 0.09, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.4), symbolic_ik.P_torso_elbowBis, marker_id + 2)
    # add_sphere(markers, 0.05, ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.4), symbolic_ik.P_torso_wristBis, marker_id + 4)
    add_sphere(markers, 0.05, ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.4), symbolic_ik.wrist_position, marker_id + 5)
    add_frame(markers, pose, marker_id + 6)

    print(f"joint: {joints}")
    if arm == "r_arm":
        real_pose = reachy.r_arm.forward_kinematics(joints)
        # real_pose_bis = reachy.r_arm.forward_kinematics(raw_joints)
    elif arm == "l_arm":
        real_pose = reachy.l_arm.forward_kinematics(joints)
        # real_pose_bis = reachy.l_arm.forward_kinematics(raw_joints)
    real_position, real_orientation = get_euler_from_homogeneous_matrix(real_pose)

    add_sphere(markers, 0.01, ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), real_position, marker_id + 3)
    # add_sphere(markers, 0.01, ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), real_pose_bis[:3, 3], marker_id + 6)

    # print(f"real pose {real_pose}")
    # print(f"goal pose {pose}")
    # print(f" Real position: {real_position}, Real orientation: {real_orientation}")
    # print(f"Goal position: {pose[0]}, Goal orientation: {pose[1]}")
    print(R.from_matrix(real_pose[:3, :3]).as_euler("xyz", degrees=True))
    print(R.from_matrix(R.from_euler("xyz", pose[1]).as_matrix()).as_euler("xyz", degrees=True))

    l2_position = np.linalg.norm(pose[0] - real_position)
    l2_orientation = np.linalg.norm(pose[1] - real_orientation)
    print(f"Position L2 norm: {l2_position}, Orientation L2 norm: {l2_orientation}")


def main_test() -> None:
    print("Trying to connect on localhost Reachy...")
    reachy = ReachySDK(host="localhost")
    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return
    reachy.turn_on()

    rclpy.init()
    node = rclpy.create_node("singularity_node")
    qos_policy = QoSProfile(depth=10)

    marker_pub = node.create_publisher(MarkerArray, "visualization_marker_array", qos_policy)
    markers = MarkerArray()
    marker_id = 0

    goal_pose = np.array([[0.38, -0.2, -0.28], [0.0, -np.pi / 2, 0.0]])
    # goal_pose = np.array([[0.38, -0.23, -0.28], [0.0, 0.0, 0.0]])

    # goal_pose = np.array([[0.001, -0.23, -0.659], [0, 0, 0]])
    # goal_pose = np.array([[0.66, -0.2, -0.], [0.0, -np.pi/2, 0.0]])

    go_to_pose(reachy, goal_pose, "r_arm", markers, marker_id)
    marker_id += 10

    goal_pose = np.array([[0.38, 0.2, -0.28], [0.0, -np.pi / 2, 0.0]])
    # goal_pose = np.array([[0.38, 0.23, -0.28], [0.0, 0.0, 0.0]])

    # goal_pose = np.array([[0.001, 0.23, -0.659], [0, 0, 0]])
    # goal_pose = np.array([[0.66, 0.2, -0.], [0.0, -np.pi/2, 0.0]])

    go_to_pose(reachy, goal_pose, "l_arm", markers, marker_id)

    marker_pub.publish(markers)


if __name__ == "__main__":
    main_test()
