import cv2
from matplotlib.style import available
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay
import math
from typing import List, Tuple, Union
import random

print("Start")

# Transformation matrix.
# For normal transformation, (not z-rotating transformation), both last row and last column must be [0, 0, a] where a > 0.
TRANSFORM_MATRIX = np.array([[1, 0, 0], [0, 1.5, 0], [0, 0, 1]])


# Index of mesh point
# nose    : 5
# top     : 10
# bottom: : 152
# right   : 50
# left    : 280


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    def limit(value, maximum):
        if value < 0:
            value = 0
        if value > maximum:
            value = maximum
        return value

    x_px = limit(math.floor(normalized_x * image_width), image_width - 1)
    y_px = limit(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def landmark_to_pixel_coordinate(image, landmark_list):
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        landmark_px = normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_cols, image_rows
        )
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    return idx_to_coordinates


def vector_to_pixel_coordinate(image, vector_list):
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, vector in enumerate(vector_list):
        landmark_px = normalized_to_pixel_coordinates(
            vector[0], vector[1], image_cols, image_rows
        )
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    return idx_to_coordinates


def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def get_faces_from_connections(connections, nodes=468):
    edges = [[False] * nodes for _ in range(nodes)]
    for edge in connections:
        s, e = edge
        assert s is not e
        edges[s][e] = True
        edges[e][s] = True

    for i in range(nodes):
        if edges[s][i] and edges[i][e]:
            break
    else:
        raise Exception("Graph is incomplete")

    visited_faces = set()
    stack = [tuple(sorted((s, i, e)))]

    def find_and_append(a, b):
        for t in range(nodes):
            if edges[a][t] and edges[t][b]:
                stack.append((a, b, t))

    while len(stack) > 0:
        current_face = tuple(sorted(stack.pop()))
        if current_face in visited_faces:
            continue
        visited_faces.add(current_face)
        s, i, e = current_face
        find_and_append(s, i)
        find_and_append(i, e)
        find_and_append(e, s)

    return list(visited_faces)


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def get_face_axis_system(landmarks):
    # x - right
    # y - top
    # z - front
    n = landmarks[5]
    t = landmarks[10]
    b = landmarks[152]
    l = landmarks[50]
    r = landmarks[280]
    o = (t + b) / 2

    x = normalize_vector(r - l)
    y = normalize_vector(t - b)
    z = normalize_vector(n - o)

    global_to_face = np.array([x, y, z])
    face_to_global = np.linalg.inv(global_to_face)

    return global_to_face, face_to_global, o


def transform_in_space(vector, matrix, axis_system):
    global_to_face, face_to_global, o = axis_system
    return (np.array(vector) - o) @ global_to_face @ matrix @ face_to_global + o


def landmark_to_matrix(landmark):
    matrix = []
    for idx, landmark in enumerate(landmark.landmark):
        vector = [landmark.x, landmark.y, landmark.z]
        matrix.append(vector)
        assert idx + 1 == len(matrix)
    return np.array(matrix)


def move_triangle(img1, img2, tri1, tri2):
    r1 = cv2.boundingRect(tri1)
    r2 = cv2.boundingRect(tri2)

    # Offset points by left top corner of the respective rectangles
    tri1_cropped = []
    tri2_cropped = []
    for i in range(0, 3):
        tri1_cropped.append(((tri1[i][0] - r1[0]), (tri1[i][1] - r1[1])))
        tri2_cropped.append(((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])))
    tri1_cropped = np.array(tri1_cropped, dtype=np.float32)
    tri2_cropped = np.array(tri2_cropped, dtype=np.float32)

    warp_mat = cv2.getAffineTransform(tri1_cropped, tri2_cropped)
    img1_cropped = img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    img2_cropped = cv2.warpAffine(
        img1_cropped,
        warp_mat,
        (r2[2], r2[3]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3))
    cv2.fillConvexPoly(mask, np.int32(tri2_cropped), (1.0, 1.0, 1.0), 16, 0)
    img2_cropped = img2_cropped * mask
    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = img2[
        r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]
    ] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = (
        img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] + img2_cropped
    )


def sort_contour_edge(contour_edges):
    contour_edges = list(contour_edges)
    current_edge = contour_edges.pop()
    sorted_edges = [current_edge]
    while len(contour_edges) > 0:
        _, p = sorted_edges[-1]
        for edge in contour_edges:
            s, e = edge
            if p == s:
                contour_edges.remove(edge)
                sorted_edges.append((s, e))
                break
            elif p == e:
                contour_edges.remove(edge)
                sorted_edges.append((e, s))
                break
        else:
            if p == sorted_edges[0][0]:
                break
            else:
                raise Exception("Graph is not closed")
    return sorted_edges


def get_edges_from_contour(contour_edges):

    circular_edges = sort_contour_edge(contour_edges)
    orderd_points = []
    for s, _ in circular_edges:
        orderd_points.append(s)

    half_length = len(orderd_points) // 2
    upper_half = orderd_points[:half_length]
    lower_half = list(reversed(orderd_points[half_length:]))
    if len(upper_half) > len(lower_half):
        upper_half, lower_half = lower_half, upper_half

    i = 0
    j = 0
    inner_edges = []
    while True:
        if i == j:
            if j + 1 >= len(upper_half):
                break
            a = upper_half[i]
            b = lower_half[j]
            j += 1
        else:
            if i + 1 >= len(lower_half):
                break
            a = upper_half[i]
            b = lower_half[j]
            i += 1
        inner_edges.append((a, b))
    return circular_edges + inner_edges


def get_outer_faces_from_contour(image, contour_edges, mapping):
    mapping_size = len(list(mapping))
    contour_edges = np.array(sort_contour_edge(contour_edges))
    ordered_points = list(map(lambda x: x[0], contour_edges))

    h, w, _ = image.shape
    # This points must be ordered.
    # N should not be too large.
    # Especially, N*4 < len(contour points).
    N = 4
    border_points = (
        [[(w * i) // N, 0] for i in range(N)]
        + [[w - 1, (h * i) // N] for i in range(N)]
        + [[w - 1 - (w * i) // N, h - 1] for i in range(N)]
        + [[0, h - 1 - (h * i) // N] for i in range(N)]
    )

    for i in range(len(border_points)):
        mapping[mapping_size + i] = border_points[i]
    border_points = np.array(border_points)

    def similarity(s, e, b):
        ep = np.array(mapping[e])
        sp = np.array(mapping[s])
        bp = np.array(mapping[b])
        center = (ep + sp) / 2
        direction_of_edge = ep - sp
        normal_of_edge = np.array([direction_of_edge[1], -direction_of_edge[0]])
        direction_to_point = bp - center
        # Cosine similarity
        dist = (
            normal_of_edge
            @ direction_to_point
            / (np.linalg.norm(normal_of_edge) * np.linalg.norm(direction_to_point))
        )
        return dist

    OP = ordered_points
    M = mapping_size
    i = 0  # index of contour points
    j = 0  # index of image border points
    max_similarity = 0
    for i in range(len(border_points)):
        dist = similarity(OP[0], OP[1], M + i)
        if dist > max_similarity:
            max_similarity = dist
            j = i
    j_init = j

    faces = []
    NC = len(ordered_points)
    NB = len(border_points)
    for i in range(NC):
        cur_dist = similarity(OP[i], OP[(i + 1) % NC], M + j)
        next_dist = similarity(OP[i], OP[(i + 1) % NC], M + (j + 1) % NB)
        if cur_dist > next_dist:
            faces.append((OP[i], OP[(i + 1) % NC], M + j))
        else:
            faces.append((OP[i], M + j, M + (j + 1) % NB))
            faces.append((OP[i], OP[(i + 1) % NC], M + (j + 1) % NB))
            j += 1
        j = j % NB
    if j != j_init:
        faces.append((OP[0], j, j_init))

    return faces


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

connections = (
    list(mp_face_mesh.FACEMESH_TESSELATION)
    + get_edges_from_contour(mp_face_mesh.FACEMESH_LEFT_EYE)
    + get_edges_from_contour(mp_face_mesh.FACEMESH_RIGHT_EYE)
    + get_edges_from_contour(mp_face_mesh.FACEMESH_LIPS)
)

faces = get_faces_from_connections(connections)

COLOR_LIST = [get_random_color() for _ in range(max(len(faces), 486))]

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        modified_image = image.copy()
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmark_matrix = landmark_to_matrix(face_landmarks)
                axis_system = get_face_axis_system(landmark_matrix)
                modified_landmark_matrix = transform_in_space(
                    landmark_matrix, TRANSFORM_MATRIX, axis_system
                )
                original_mapping = vector_to_pixel_coordinate(image, landmark_matrix)
                modified_mapping = vector_to_pixel_coordinate(
                    image, modified_landmark_matrix
                )

                rf1 = get_outer_faces_from_contour(
                    image, mp_face_mesh.FACEMESH_FACE_OVAL, original_mapping
                )
                rf2 = get_outer_faces_from_contour(
                    image, mp_face_mesh.FACEMESH_FACE_OVAL, modified_mapping
                )

                full_faces = faces + rf1 + rf2

                for i in range(len(full_faces)):
                    try:
                        original_points = np.array(
                            list(
                                map(
                                    lambda index: original_mapping[index], full_faces[i]
                                )
                            )
                        )
                        modified_points = np.array(
                            list(
                                map(
                                    lambda index: modified_mapping[index], full_faces[i]
                                )
                            )
                        )
                        move_triangle(
                            image, modified_image, original_points, modified_points
                        )

                        # a, b, c = modified_points
                        # cv2.line(modified_image, a, b, (0, 0, 0), 1)
                        # cv2.line(modified_image, a, c, (0, 0, 0), 1)
                        # cv2.line(modified_image, c, b, (0, 0, 0), 1)

                    except KeyError:
                        continue

        cv2.imshow("MediaPipe Face Mesh", modified_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
