import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import tf_transformations
import math


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    if np.all(np.isclose(vec1, vec2)):
        return np.eye(3)
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def circle(center, radius, normal_vector, intervalles, color):
    theta = []
    # intervalles = [[2.172615051230774, 5.549674318226633]]
    for intervalle in intervalles:
        print(intervalle)
        angle = np.linspace(intervalle[0], intervalle[1], 100)
        print("test")
        for a in angle:
            theta.append(a)
    print(len(theta))

    # Paramètres paramétriques pour le cercle en 3D
    y = radius * np.cos(theta)
    z = radius * np.sin(theta)
    x = np.zeros(len(theta))

    # ax.plot(x,y,z, 'r')

    Rmat = rotation_matrix_from_vectors(np.array([1, 0, 0]), normal_vector)
    vect = np.dot(Rmat, np.array([0, 0, 1]))
    # print(vect)
    Tmat = np.array(
        [
            [Rmat[0][0], Rmat[0][1], Rmat[0][2], center[0]],
            [Rmat[1][0], Rmat[1][1], Rmat[1][2], center[1]],
            [Rmat[2][0], Rmat[2][1], Rmat[2][2], center[2]],
            [0, 0, 0, 1],
        ]
    )

    ax.plot([center[0], center[0] + vect[0]], [center[1], center[1] + vect[1]], [center[2], center[2] + vect[2]], color)
    p = [0, 0, 2, 1]
    p2 = np.dot(Tmat, p)
    # print(p)
    x2 = np.zeros(len(theta))
    y2 = np.zeros(len(theta))
    z2 = np.zeros(len(theta))
    for k in range(len(theta)):
        p = [x[k], y[k], z[k], 1]
        p2 = np.dot(Tmat, p)
        x2[k] = p2[0]
        y2[k] = p2[1]
        z2[k] = p2[2]

    ax.plot(center[0], center[1], center[2], "o", color=color)
    ax.plot(x2, y2, z2, color)
    ax.plot([0, center[0]], [0, center[1]], [0, center[2]], color)
    ax.plot(x2[0], y2[0], z2[0], "o", color=color)

    ax.axes.set_xlim3d(left=-3, right=3)
    ax.axes.set_ylim3d(bottom=-3, top=3)
    ax.axes.set_zlim3d(bottom=-3, top=3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.legend()

    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
    x = np.cos(u) * np.sin(v) * 3
    y = np.sin(u) * np.sin(v) * 3
    z = np.cos(v) * 3
    ax.plot_wireframe(x, y, z, color="r", alpha=0.2)
    ax.plot(0, 0, 0, "ro")

    # show vect


def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def intersection_point(v1, p01, v2, p02):
    A = np.vstack((v1, -v2)).T
    b = p02 - p01
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Vérifier si les vecteurs sont colinéaires (pas d'intersection unique)
    if np.all(np.isclose(params, params[0])):
        return None

    intersection = v1 * params[0] + p01
    return intersection


def points_of_nearest_approach(p1, n1, p2, n2):
    v = np.cross(n1, n2)
    A = np.array([n1, -n2, v]).T
    b = p2 - p1
    s, t, _ = np.linalg.lstsq(A, b, rcond=None)[0]
    q1 = p1 + s * (np.cross(v, n1))
    q2 = p2 + t * (np.cross(v, n2))
    vect1 = np.cross(v, n1)
    v = v / np.linalg.norm(v)
    ax.plot([center1[0], center1[0] + vect1[0]], [center1[1], center1[1] + vect1[1]], [center1[2], center1[2] + vect1[2]], "r")
    vect2 = np.cross(v, n2)
    ax.plot([center2[0], center2[0] + vect2[0]], [center2[1], center2[1] + vect2[1]], [center2[2], center2[2] + vect2[2]], "r")
    q = intersection_point(vect1, p1, vect2, p2)
    # print(vect1, vect2)
    if q is not None:
        ax.plot(q[0], q[1], q[2], "ro")
        ax.plot([q[0] - 2 * v[0], q[0] + v[0] * 2], [q[1] - 2 * v[1], q[1] + v[1] * 2], [q[2] - 2 * v[2], q[2] + 2 * v[2]], "r")
    return q, v


def planar_point_in_circle(point, center, radius):
    return distance(point, center) < radius


def are_circles_linked(p1, n1, r1, p2, n2, r2):
    # get data in sphere frame
    [x, y, z] = p1
    d = np.linalg.norm([x, y, z])
    Mrot = R.from_euler("xyz", [0.0, -math.asin(z / d), math.atan2(y, x)]).as_matrix()
    Trot = np.array(
        [
            [Mrot[0][0], Mrot[0][1], Mrot[0][2], p1[0]],
            [Mrot[1][0], Mrot[1][1], Mrot[1][2], p1[1]],
            [Mrot[2][0], Mrot[2][1], Mrot[2][2], p1[2]],
            [0, 0, 0, 1],
        ]
    )
    Trot_t = Trot.T
    Rmat = rotation_matrix_from_vectors(np.array([1, 0, 0]), n2)
    # vect = np.dot(Rmat, np.array([0, 0, 1]))
    # print(vect)
    Tmat = np.array(
        [
            [Rmat[0][0], Rmat[0][1], Rmat[0][2], p2[0]],
            [Rmat[1][0], Rmat[1][1], Rmat[1][2], p2[1]],
            [Rmat[2][0], Rmat[2][1], Rmat[2][2], p2[2]],
            [0, 0, 0, 1],
        ]
    )
    Trot_t = Tmat.T
    center1 = np.array([p1[0], p1[1], p1[2], 1])
    center2 = np.array([p2[0], p2[1], p2[2], 1])
    center1_in_sphere_frame = np.dot(Trot_t, center1)
    print(center1_in_sphere_frame)
    center2_in_sphere_frame = np.dot(Trot_t, center2)
    print(center2_in_sphere_frame)
    ax.plot(center2_in_sphere_frame[0], center2_in_sphere_frame[1], center2_in_sphere_frame[2], "ro")

    if np.all(n1 == -n2) or np.all(n1 == n2):
        print("concurrent or parallel")
        return center2_in_sphere_frame[0] < center1_in_sphere_frame[0]
    else:
        # Find the line of intersection of the planes
        q, v = points_of_nearest_approach(p1, n1, p2, n2)
        if q is None:
            return center2_in_sphere_frame[0] < center1_in_sphere_frame[0]
        points = intersection_circle_line_3d_vd(p1, r1, v, q)
        if points is None:
            return center2_in_sphere_frame[0] < center1_in_sphere_frame[0]
        for point in points:
            plt.plot(point[0], point[1], point[2], "ro")

        # TODO mettres les points dans le repere de la sphere et definir le bon intervalle d'angles
        intervalle = [[0, 2 * np.pi]]
        if len(points == 2):
            point1 = [points[0][0], points[0][1], points[0][2], 1]
            point2 = [points[1][0], points[1][1], points[1][2], 1]
            point1_in_sphere_frame = np.dot(Trot_t, point1)
            point2_in_sphere_frame = np.dot(Trot_t, point2)
            print("--")
            print(point1_in_sphere_frame)
            print(point2_in_sphere_frame)
            angle1 = math.atan2(point1_in_sphere_frame[2], point1_in_sphere_frame[1])
            angle2 = math.atan2(point2_in_sphere_frame[2], point2_in_sphere_frame[1])
            if angle1 < 0:
                angle1 = angle1 + 2 * np.pi
            if angle2 < 0:
                angle2 = angle2 + 2 * np.pi

            [angle1, angle2] = sorted([angle1, angle2])
            print(angle1, angle2)
            angle_test = (angle1 + angle2) / 2
            P = np.array([0, math.cos(angle_test), math.sin(angle_test), 1])
            P = np.dot(Tmat, P)
            ax.plot(P[0], P[1], P[2], "ro")
            T = Trot.T
            P = np.dot(T, P)
            print(P)
            center1_in_frame = np.dot(T, center1)
            print(center1_in_frame)
            print(P[0] > center1_in_frame[0])
            if P[0] > center1_in_frame[0]:
                intervalle = [[angle1, angle2]]
            else:
                intervalle = [[0, angle1], [angle2, 2 * np.pi]]

            print(intervalle)

            print(angle1, angle2)
            circle(p2, r2, n2, intervalle, "g")

        return True


def intersection_circle_line_3d_vd(center, radius, direction, point_on_line):
    # Équation du cercle : (x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2 = r^2
    # Équation de la droite : p(t) = point_on_line + t * direction

    a = np.dot(direction, direction)  # Coefficient devant t^2
    b = 2 * np.dot(direction, point_on_line - center)  # Coefficient devant t
    c = np.dot(point_on_line - center, point_on_line - center) - radius**2  # Terme constant

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # Pas d'intersection
        return None
    elif discriminant == 0:
        # Une seule intersection
        t = -b / (2 * a)
        intersection = point_on_line + t * direction
        return np.array([intersection])
    else:
        # Deux intersections
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        intersection1 = point_on_line + t1 * direction
        intersection2 = point_on_line + t2 * direction
        return np.vstack((intersection1, intersection2))


# Exemple d'utilisation avec un vecteur normal au cercle
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot([0, 0.5], [0, 0], [0, 0], "r")
ax.plot([0, 0], [0, 0.5], [0, 0], "g")
ax.plot([0, 0], [0, 0], [0, 0.5], "b")


centers = []
radiuss = []
normal_vectors = []
colors = []

sphere_radius = 3
vect = [2, -1, 3]
radius1 = 2
if radius1 >= sphere_radius:
    center1 = np.array([0, 0, 0])
else:
    center1 = np.array(vect / np.linalg.norm(vect) * np.sqrt(sphere_radius**2 - radius1**2))
normal_vector1 = np.array(vect / np.linalg.norm(vect))

vect = [2, -1, 3]
radius2 = 1
if radius2 >= sphere_radius:
    center2 = np.array([0, 0, 0])
else:
    center2 = np.array(vect / np.linalg.norm(vect) * np.sqrt(sphere_radius**2 - radius2**2))
normal_vector2 = np.array(vect / np.linalg.norm(vect))


linked = are_circles_linked(center1, normal_vector1, radius1, center2, normal_vector2, radius2)
print("Are circles linked?", linked)

# circle(centers, radiuss, normal_vectors, colors)
circle(center1, radius1, normal_vector1, [[0, 2 * np.pi]], "b")
circle(center2, radius2, normal_vector2, [[0, 2 * np.pi]], "g")
# circle([0,0,0], 2, [1,0,0],[0, 2*np.pi], 'g')
print(normal_vector1)
print(center1)

# print(roll_axe)
plt.show()
