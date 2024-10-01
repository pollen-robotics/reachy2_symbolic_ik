import numpy as np
import numpy.typing as npt
import rclpy
from rclpy.qos import QoSProfile
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import make_homogenous_matrix_from_rotation_matrix

# def singularity_position_by_offset(
#     shoulder_orientation_offset: npt.NDArray[np.float64], upper_arm_size: float = 0.28, arm: str = "r"
# ) -> npt.NDArray[np.float64]:
#     if arm == "r":
#         shoulder_position = [0, -0.2, 0, 1]
#         singularity_position = [0, -0.28, 0, 1]
#     else:
#         shoulder_position = [0, 0.2, 0, 1]
#         singularity_position = [0, 0.28, 0, 1]
#         # singularity_position[1] *= -1
#     rotation = R.from_euler("xyz", np.radians(shoulder_orientation_offset)).as_matrix()
#     homogeneous_matrix = [
#         [rotation[0][0], rotation[0][1], rotation[0][2], 0],
#         [rotation[1][0], rotation[1][1], rotation[1][2], shoulder_position[1]],
#         [rotation[2][0], rotation[2][1], rotation[2][2], 0],
#         [0, 0, 0, 1],
#     ]
#     # homogeneous_matrix_transpose = np.transpose(homogeneous_matrix)
#     singularity_position = np.dot(homogeneous_matrix, singularity_position)
#     # singularity_position[1] += -0.2
#     return np.array(singularity_position)


def add_sphere(markers: Marker, size: float, color: ColorRGBA, position: npt.NDArray[np.float64], index: int = 0) -> None:
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

    markers.markers.append(
        marker
        # create_sphere([0, 0, 0], ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), 0.03)
    )


def add_plane(
    markers: Marker,
    size_x: float,
    size_y: float,
    color: ColorRGBA,
    position: npt.NDArray[np.float64],
    index: int = 0,
    quaternion: npt.NDArray[np.float64] = R.from_euler("xyz", [0, 0, 0]).as_quat(),
) -> None:
    marker = Marker()
    marker.header.frame_id = "torso"
    marker.type = marker.CUBE
    marker.action = marker.ADD
    marker.scale.x = size_x
    marker.scale.y = size_y
    marker.scale.z = 0.0001
    marker.color.a = color.a
    marker.color.r = color.r
    marker.color.g = color.g
    marker.color.b = color.b

    # rotation = R.from_euler("xyz", [np.pi/2, 0, 0]).as_quat()
    marker.pose.orientation.x = quaternion[0]
    marker.pose.orientation.y = quaternion[1]
    marker.pose.orientation.z = quaternion[2]
    marker.pose.orientation.w = quaternion[3]
    # marker.pose.orientation.w = 1.0
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.id = index

    markers.markers.append(
        marker
        # create_sphere([0, 0, 0], ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), 0.03)
    )


def add_circle(
    markers: Marker,
    radius: float,
    color: ColorRGBA,
    position: npt.NDArray[np.float64],
    index: int = 0,
    quaternion: npt.NDArray[np.float64] = R.from_euler("xyz", [0, 0, 0]).as_quat(),
) -> None:
    marker = Marker()
    marker.header.frame_id = "torso"
    marker.type = marker.CYLINDER
    marker.action = marker.ADD
    marker.scale.x = radius
    marker.scale.y = radius
    marker.scale.z = 0.0001
    marker.color.a = color.a
    marker.color.r = color.r
    marker.color.g = color.g
    marker.color.b = color.b

    # rotation = R.from_euler("xyz", [np.pi/2, 0, 0]).as_quat()
    marker.pose.orientation.x = quaternion[0]
    marker.pose.orientation.y = quaternion[1]
    marker.pose.orientation.z = quaternion[2]
    marker.pose.orientation.w = quaternion[3]

    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.id = index

    markers.markers.append(marker)


# def make_elbow_projection(
#     elbow_position: npt.NDArray[np.float64],
#     shoulder_position: npt.NDArray[np.float64],
#     upper_arm_size: float,
#     singularity_offset: float,
#     singularity_limit_coeff: float,
#     elbow_singularity_position: npt.NDArray[np.float64],
# ) -> npt.NDArray[np.float64]:
#     alpha = np.arctan2(-singularity_limit_coeff, 1)
#     M_limits = R.from_euler("xyz", [0, alpha, 0]).as_matrix()
#     P_limits = [
#         elbow_singularity_position[0],
#         elbow_singularity_position[1],
#         elbow_singularity_position[2] - singularity_offset,
#         1,
#     ]
#     T_limits = make_homogenous_matrix_from_rotation_matrix(P_limits, M_limits)

#     # get normal vector
#     n1 = np.array([1, 0, 0, 1])
#     n2 = np.array([0, 1, 0, 1])
#     n1 = np.dot(T_limits, n1)
#     n2 = np.dot(T_limits, n2)
#     v1 = n1 - P_limits
#     v2 = n2 - P_limits
#     v3 = np.cross(v1[:3], v2[:3])
#     v3 = v3 / np.linalg.norm(v3)

#     projected_center = get_projection_point(v3, P_limits[:3], shoulder_position)
#     radius = np.sqrt(upper_arm_size**2 - np.linalg.norm(shoulder_position - projected_center) ** 2)

#     projected_elbow = get_projection_point(v3, P_limits[:3], elbow_position[:3])
#     V_center_projection = projected_elbow - projected_center
#     new_elbow_position = projected_center + radius * (V_center_projection / np.linalg.norm(V_center_projection))
#     return new_elbow_position


# def get_projection_point(normal_vector, plane_point, point):
#     v = point - plane_point
#     dist = np.dot(v, normal_vector)
#     # print(dist)
#     projected_point = point - dist * normal_vector
#     return projected_point


if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("singularity_node")
    qos_policy = QoSProfile(depth=10)

    marker_pub = node.create_publisher(MarkerArray, "visualization_marker_array", qos_policy)
    markers = MarkerArray()

    ik_r = SymbolicIK("r_arm")
    ik_l = SymbolicIK("l_arm")

    alpha = np.arctan2(-ik_r.singularity_limit_coeff, 1)
    size = 0.6

    index = 0

    for ik in [ik_r, ik_l]:
        rotation = R.from_euler("xyz", [0, 0, ik.shoulder_orientation_offset[2]], degrees=True)
        rotation2 = R.from_euler("xyz", [0, alpha, 0])
        rotation3 = rotation * rotation2
        rotation3 = rotation3.as_quat()

        if ik.arm == "r_arm":
            side = -1
        else:
            side = 1
        x = np.cos(alpha) * size / 2
        y = size / 2 * side
        z = ik.singularity_limit_coeff * x + ik.wrist_singularity_position[2] - ik.singularity_offset
        P = np.array([x, y, z])

        M_torso_shoulderYaw = R.from_euler("xyz", [0, 0, ik.shoulder_orientation_offset[2]], degrees=True).as_matrix()
        T_shoulderYaw_torso = make_homogenous_matrix_from_rotation_matrix(ik.shoulder_position, M_torso_shoulderYaw)
        P_torso = np.dot(T_shoulderYaw_torso, np.array([x, y, z, 1]))

        P = np.array([-size / 2, size / 2 * side, ik.wrist_singularity_position[2] - ik.singularity_offset, 1])
        P = np.dot(T_shoulderYaw_torso, P)

        M_torso_shoulderYaw_quat = R.from_euler("xyz", [0, 0, ik.shoulder_orientation_offset[2]], degrees=True).as_quat()

        add_plane(markers, size, size, ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.3), P_torso, index, rotation3)
        index += 1
        add_plane(markers, size, size, ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.3), P, index, M_torso_shoulderYaw_quat)
        index += 1

        P_elbow1 = [P_torso[0], P_torso[1], P_torso[2] - ik.wrist_singularity_position[2] + ik.elbow_singularity_position[2]]
        P_elbow2 = [P[0], P[1], P[2] - ik.wrist_singularity_position[2] + ik.elbow_singularity_position[2]]

        # add_plane(markers, size, size, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3), P_elbow1, index, rotation3)
        # index += 1
        # add_plane(markers, size, size, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3), P_elbow2, index, M_torso_shoulderYaw_quat)
        # index += 1

        # p1 = np.array(
        #     [
        #         ik.elbow_singularity_position[0] - size / 2,
        #         ik.elbow_singularity_position[1],
        #         ik.elbow_singularity_position[2] - ik.singularity_offset,
        #         1,
        #     ]
        # )
        # add_plane(
        #     markers,
        #     size,
        #     size,
        #     ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3),
        #     p1,
        #     index,
        # )
        # index += 1

        # y = ik.elbow_singularity_position[1]

        # x = size / 2 / np.sqrt(1 + ik.singularity_limit_coeff**2) + ik.elbow_singularity_position[0]
        # z = (
        #     size / 2 * ik.singularity_limit_coeff / np.sqrt(1 + ik.singularity_limit_coeff**2)
        #     + ik.elbow_singularity_position[2]
        #     - ik.singularity_offset
        # )
        # p2 = np.array([x, y, z])
        p2 = np.array(
            [
                ik.elbow_singularity_position[0],
                ik.elbow_singularity_position[1],
                ik.elbow_singularity_position[2] - ik.singularity_offset,
                1,
            ]
        )

        # elbow_position = np.array([0.2, 0.3 * side, 0.1, 1])
        # add_sphere(markers, 0.01, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3), elbow_position, index)
        # index += 1

        # new_elbow_position = make_elbow_projection(
        #     elbow_position,
        #     ik.shoulder_position,
        #     ik.upper_arm_size,
        #     ik.singularity_offset,
        #     ik.singularity_limit_coeff,
        #     ik.elbow_singularity_position,
        # )
        # add_sphere(markers, 0.01, ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3), new_elbow_position, index)
        # index += 1

        # T_r2 = make_homogenous_matrix_from_rotation_matrix(p2[:3], rotation2.as_matrix())
        # pp = [0,0,0,1]
        # pp = np.dot(T_r2, pp)
        # ppp = [1, 0.5, 0, 1]
        # ppp = np.dot(T_r2, ppp)
        # print(pp)
        # print(ppp)
        # print(np.sqrt(ppp[0]**2 + ppp[1]**2+ ppp[2]**2))
        # P_shoulder_position = np.dot(T_r2, [ik.shoulder_position[0], ik.shoulder_position[1], ik.shoulder_position[2], 1])
        # print(f"radius {radius}")

        # n1 = np.array([0.1, 0, 0, 1])
        # n2 = np.array([0, 0.1, 0, 1])
        # n1 = np.dot(T_r2, n1)
        # n2 = np.dot(T_r2, n2)
        # v1 = n1 - p2
        # v2 = n2 - p2
        # v3 = np.cross(v1[:3], v2[:3])
        # v3 = v3 / np.linalg.norm(v3)
        # n3 = v3 + p2[:3]
        # print(f"n3 {n3}")

        # projected_center = get_projection_point(v3, p2[:3], ik.shoulder_position)
        # radius = np.sqrt(ik.upper_arm_size**2 - np.linalg.norm( ik.shoulder_position - projected_center[:3])**2)

        # random_point = np.array([-0.2, 0.3*side, 0.1, 1])

        # projected__point = get_projection_point(v3, p2[:3], random_point[:3])

        # add_sphere(markers, 0.01, ColorRGBA(r=.0, g=1.0, b=.0, a=0.3), projected_center, index)
        # index += 1

        # add_sphere(markers, 0.01, ColorRGBA(r=1.0, g=.0, b=.0, a=0.3), random_point, index)
        # index += 1

        # add_sphere(markers, 0.01, ColorRGBA(r=.0, g=1.0, b=.0, a=0.3), projected__point, index)
        # index += 1

        # vector = projected__point - projected_center
        # projected_point_on_circle = projected_center + radius* (vector/np.linalg.norm(vector))
        # print(f"test {np.linalg.norm(projected_point_on_circle-projected_center)}")
        # print(f"radius {radius}")

        # add_sphere(markers, 0.01, ColorRGBA(r=1.0, g=1.0, b=.0, a=0.3), projected_point_on_circle, index)
        # index += 1

        # add_circle(markers, radius*2, ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.3), projected_center, index, rotation2.as_quat())
        # index += 1

        # v = random_point - p2
        # dist = np.dot(v[:3], v3)
        # print(dist)
        # projection = random_point[:3] - dist * v3

        # add_sphere(markers, 0.01, ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.3), p2, index)
        # index += 1
        # add_sphere(markers, 0.01, ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.3), n1, index)
        # index += 1
        # add_sphere(markers, 0.01, ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.3), n2, index)
        # index += 1
        # add_sphere(markers, 0.01, ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.3), n3, index)
        # index += 1

        add_plane(
            markers,
            size * 2,
            size * 1.5,
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3),
            p2,
            index,
            rotation2.as_quat(),
        )
        index += 1

        add_sphere(markers, 0.1, ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.3), ik.elbow_singularity_position, index)
        index += 1
        add_sphere(markers, 0.01, ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0), ik.elbow_singularity_position, index)
        index += 1

        # random_point = np.array([0.2, 0.3*side, -0.3, 1])
        # add_sphere(markers, 0.01, ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0), random_point, index)
        # index += 1
        # random_point = np.dot(T_r2, random_point)
        # add_sphere(markers, 0.01, ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0), random_point, index)
        # index += 1

    print(f"singularity position {ik_r.elbow_singularity_position}")

    add_sphere(markers, 0.56, ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3), ik_r.shoulder_position, 30)

    marker_pub.publish(markers)
