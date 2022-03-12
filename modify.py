import cv2
from matplotlib.style import available
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay
import math
from typing import List, Tuple, Union
import random

print('Start')

TRANSFORM_MATRIX = np.array([
  [1,0,0],
  [0,1.5,0],
  [0,0,1]
])
# Vertical scaling matrix


# Index of mesh point
# nose    : 5
# top     : 10
# bottom: : 152 
# right   : 50
# left    : 280

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def landmark_to_pixel_coordinate(image,landmark_list):
  image_rows, image_cols, _ = image.shape
  idx_to_coordinates = {}
  for idx, landmark in enumerate(landmark_list.landmark):
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px
  return idx_to_coordinates

def vector_to_pixel_coordinate(image,vector_list):
  image_rows, image_cols, _ = image.shape
  idx_to_coordinates = {}
  for idx, vector in enumerate(vector_list):
    landmark_px = _normalized_to_pixel_coordinates(vector[0],vector[1], image_cols, image_rows)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px
  return idx_to_coordinates

def get_random_color():
  return (random.randint(0,255),random.randint(0,255),random.randint(0,255))

def get_faces_from_connections(connections,nodes=468):
  edges = [[False]*nodes for _ in range(nodes)]
  for edge in connections:
    s,e = edge
    assert s is not e
    edges[s][e] = True
    edges[e][s] = True

  for i in range(nodes):
    if edges[s][i] and edges[i][e]:
      break
  else:
    raise Exception("Graph is incomplete")

  visited_faces = set()
  stack = [tuple(sorted((s,i,e)))]

  def find_and_append(a,b):
    for t in range(nodes):
      if edges[a][t] and edges[t][b]:
        stack.append((a,b,t))

  while len(stack)>0:
    current_face = tuple(sorted(stack.pop()))
    if current_face in visited_faces:
      continue
    visited_faces.add(current_face)
    s,i,e = current_face
    find_and_append(s,i)
    find_and_append(i,e)
    find_and_append(e,s)

  return list(visited_faces)

def normalize_vector(vector):
  return vector/np.linalg.norm(vector)

def get_face_axis_system(landmarks):
  # x - right
  # y - top
  # z - front
  n = landmarks[5]
  t = landmarks[10]
  b = landmarks[152]
  l = landmarks[50]
  r = landmarks[280]
  o = (t+b)/2

  x = normalize_vector(r-l)
  y = normalize_vector(t-b)
  z = normalize_vector(n-o)

  global_to_face = np.array([x,y,z])
  face_to_global = np.linalg.inv(global_to_face)

  return global_to_face,face_to_global,o

def transform_in_space(vector, matrix, axis_system):
  global_to_face,face_to_global,o = axis_system
  return (np.array(vector)-o)@global_to_face@matrix@face_to_global+o

def landmark_to_matrix(landmark):
  matrix = []
  for idx, landmark in enumerate(landmark.landmark):
    vector = [landmark.x, landmark.y, landmark.z]
    matrix.append(vector)
    assert idx+1 == len(matrix)
  return np.array(matrix)

def move_triangle(img1, img2, tri1, tri2):

  r1 = cv2.boundingRect(tri1)
  r2 = cv2.boundingRect(tri2)

  # Offset points by left top corner of the respective rectangles
  tri1Cropped = []
  tri2Cropped = []
  for i in range(0, 3):
      tri1Cropped.append(((tri1[i][0] - r1[0]),(tri1[i][1] - r1[1])))
      tri2Cropped.append(((tri2[i][0] - r2[0]),(tri2[i][1] - r2[1])))
  tri1Cropped = np.array(tri1Cropped, dtype=np.float32)
  tri2Cropped = np.array(tri2Cropped, dtype=np.float32)

  warpMat = cv2.getAffineTransform(tri1Cropped, tri2Cropped)
  img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
  img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

  # Get mask by filling triangle
  mask = np.zeros((r2[3], r2[2], 3))
  cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);
  img2Cropped = img2Cropped * mask
  img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
  img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

faces = get_faces_from_connections(mp_face_mesh.FACEMESH_TESSELATION)

COLOR_LIST = [get_random_color() for _ in range(max(len(faces),486))]

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

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
        modified_landmark_matrix = transform_in_space(landmark_matrix,TRANSFORM_MATRIX,axis_system)
        original_mapping = vector_to_pixel_coordinate(image, landmark_matrix)
        modified_mapping = vector_to_pixel_coordinate(image, modified_landmark_matrix)

        try:
          for i in range(len(faces)):
            original_points = np.array(list(map(lambda index:original_mapping[index],faces[i])))
            modified_points = np.array(list(map(lambda index:modified_mapping[index],faces[i])))   
            move_triangle(image,modified_image,original_points,modified_points)
        except KeyError:
          continue

    cv2.imshow('MediaPipe Face Mesh', modified_image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()