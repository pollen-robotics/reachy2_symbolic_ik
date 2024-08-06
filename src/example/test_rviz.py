import numpy as np
import numpy.typing as npt
import rclpy
from rclpy.qos import QoSProfile
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def singularity_position_by_offset(
    shoulder_orientation_offset: npt.NDArray[np.float64], upper_arm_size: float = 0.28, arm: str = "r"
) -> npt.NDArray[np.float64]:
    if arm == "r":
        shoulder_position = [0, -0.2, 0, 1]
        singularity_position = [0, -0.28, 0, 1]
    else:
        shoulder_position = [0, 0.2, 0, 1]
        singularity_position = [0, 0.28, 0, 1]
        # singularity_position[1] *= -1
    rotation = R.from_euler("xyz", np.radians(shoulder_orientation_offset)).as_matrix()
    homogeneous_matrix = [
        [rotation[0][0], rotation[0][1], rotation[0][2], 0],
        [rotation[1][0], rotation[1][1], rotation[1][2], shoulder_position[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], 0],
        [0, 0, 0, 1],
    ]
    # homogeneous_matrix_transpose = np.transpose(homogeneous_matrix)
    singularity_position = np.dot(homogeneous_matrix, singularity_position)
    # singularity_position[1] += -0.2
    return np.array(singularity_position)


def add_sphere(marker: Marker, size: float, color: ColorRGBA, position: npt.NDArray[np.float64], index: int = 0) -> None:
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


if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("singularity_node")
    qos_policy = QoSProfile(depth=10)

    marker_pub = node.create_publisher(MarkerArray, "visualization_marker_array", qos_policy)
    markers = MarkerArray()
    add_sphere(markers, 0.1, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.4), np.array([0.3, 0.0, 0.0]))
    add_sphere(markers, 0.1, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3), singularity_position_by_offset(np.array([10, 0, 15])), 0)
    add_sphere(markers, 0.01, ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), singularity_position_by_offset(np.array([10, 0, 15])), 4)
    add_sphere(
        markers, 0.1, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3), singularity_position_by_offset(np.array([-10, 0, -15]), arm="l"), 8
    )
    add_sphere(
        markers,
        0.01,
        ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
        singularity_position_by_offset(np.array([-10, 0, -15]), arm="l"),
        9,
    )

    # shoulder_orientation_offset = [0, 0, 0]
    # shoulder_position = [0, -0.2, 0]
    # add_sphere(markers, 0.56, ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.3), [0.0, -0.2, 0.0], 1)
    # add_sphere(markers, 0.56, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3), np.array([0.45, -0.2, 0.0]), 10)
    # add_sphere(markers, 0.05, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8), np.array([0.45, -0.2, 0.0]), 9)

    # add_sphere(markers, 0.1, ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.3), singularity_position_by_offset([0, 0, 0]), 1)
    # add_sphere(markers, 0.01, ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.), singularity_position_by_offset([0, 0, 0]), 5)
    # add_sphere(markers, 0.1, ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.3), singularity_position_by_offset([0, 0, 0], arm="l"), 10)
    # add_sphere(markers, 0.01, ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.), singularity_position_by_offset([0, 0, 0], arm="l"), 11)

    # add_sphere(markers, 0.1, ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3), singularity_position_by_offset([-10, 0, -5]), 2)
    # add_sphere(markers, 0.01, ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.), singularity_position_by_offset([-10, 0, -5]), 6)
    # add_sphere(markers, 0.1, ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3), singularity_position_by_offset([10, 0, 5], arm="l"), 12)
    # add_sphere(markers, 0.01, ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.), singularity_position_by_offset([10, 0, 5], arm="l"), 13)

    # add_sphere(markers, 0.1, ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.3), singularity_position_by_offset([0, 0, -5]), 3)
    # add_sphere(markers, 0.01, ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0), singularity_position_by_offset([0, 0, -5]), 7)
    # add_sphere(markers, 0.1, ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.3), singularity_position_by_offset([0, 0, 5], arm="l"), 14)
    # add_sphere(markers, 0.01, ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0), singularity_position_by_offset([0, 0, 5], arm="l"), 15)

    # marker = Marker()
    # marker.header.frame_id = "torso"
    # marker.type = marker.SPHERE
    # marker.action = marker.ADD
    # marker.scale.x = 0.1
    # marker.scale.y = 0.1
    # marker.scale.z = 0.1
    # marker.color.a = 0.4
    # marker.color.r = 1.0
    # marker.color.g = 0.0
    # marker.color.b = 0.0
    # marker.pose.orientation.w = 1.0
    # marker.pose.position.x = 0.3
    # marker.pose.position.y = 0.
    # marker.pose.position.z = 0.

    # markers.markers.append(
    #     marker
    #     # create_sphere([0, 0, 0], ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), 0.03)
    # )
    # for m in markers.markers:
    #    m.id = id
    #    id += 1

    marker_pub.publish(markers)


# topic = 'visualization_marker_array'
# publisher = rospy.Publisher(topic, MarkerArray)

# rospy.init_node('register')

# markerArray = MarkerArray()

# # ... here I get the data I want to plot into a vector called trans

# marker = Marker()
# marker.header.frame_id = "/neck"
# marker.type = marker.SPHERE
# marker.action = marker.ADD
# marker.scale.x = 0.2
# marker.scale.y = 0.2
# marker.scale.z = 0.2
# marker.color.a = 1.0
# marker.pose.orientation.w = 1.0
# marker.pose.position.x = 0
# marker.pose.position.y = 0
# marker.pose.position.z = 0
# # We add the new marker to the MarkerArray, removing the oldest marker from it when necessary
# # if(count > MARKERS_MAX):
# # markerArray.markers.pop(0)
# # else:
# # count += 1
# markerArray.markers.append(marker)


# # Publish the MarkerArray
# publisher.publish(markerArray)
