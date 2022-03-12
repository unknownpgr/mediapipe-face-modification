# Mediapipe Face Modification

**CAUTION: 본 프로젝트는 python 3.9 이상에서는 작동하지 않음. python 3.8에서 테스트됨.**

[MediaPipe](https://github.com/google/mediapipe) & OpenCV 기반의 얼굴 변조 스크립트를 작성하였다. `modify.py` 스크립트는 mesh를 바탕으로 다음 과정을 수행한다.

0. 영상 처리를 시작하기 전 mesh 각 포인트의 connection 정보를 바탕으로 mesh를 이루는 각 face(=3개의 edge로 구성되는 한 개의 면)정보를 미리 계산한다. 일반적인 (그래프와는 다르게) mesh는 오직 삼각형으로만 이루어진 plannar graph이므로 embedding이 유일하여 이것이 가능하다.

1. 영상을 입력받아 각 프레임별로 MediaPipe를 사용하여 얼굴 mesh(468개의 landmark로 이루어진)를 얻는다.

2. 얼굴을 기준으로 한 직교좌표계 및 얼굴의 중점(O)을 계산한다. 화면 기준으로 좌측이 x방향, 위쪽이 y방향, 화면에서 나오는 방향이 z방향이다.

3. 이로부터 전역 좌표계에서 얼굴 기준 좌표계로 바꾸는 변환 `T1`, 얼굴 기준 좌표계에서 전역 좌표계로 변환하는 변환 `T2`를 계산한다.

4. 어떤 3x3 matrix로 주어지는 linear transform `M` 에 대하여, 얼굴 mesh의 각 랜드마크L (3차원 벡터)에 대해 `(L-O) × T1 × M × T2 + O` 를 계산하여 mesh에 `M` 연산을 적용한다. 실제로는 `L`에 각 랜드마크 대신 전체 랜드마크가 포함된 `(468,3)` 크기의 행렬을 대입하며, numpy의 broadcasting을 통해 한 번에 모든 랜드마크를 계산한다.

5. 사본 이미지를 만든 후, affine transform을 이용하여 mesh의 각 face를 원본에서 사본으로 옮긴다.

# 추가 보완할 부분

- 현재는 mediapipe에서 눈 부분이 mesh에서 제외되어있어 변형한 후의 얼굴이 어색하다. 그러나 이를 포함하는 것은 매우 간단할 것이다. 눈 부분의 mesh 정보에 대해서는 mediapipe 라이브러리의 `python/solutions/face_mesh.py` 및 `face_mesh_connections.py`를 참조하라.

- 추가로 현재는 오직 얼굴 mesh만 변형 부분에 포함되므로 얼굴 외부의 포인트들은 그대로 남아 있고, 따라서 얼굴을 축소하는 경우에는 기존의 얼굴이 그대로 보이게 된다. 그러므로 mesh를 얼굴 외부로 확장할 필요가 있다. 그러기 위해서는 얼굴 mesh의 bouding box나 이미지 자체의 테두리 부분을 mesh 포인트에 포함시킨 후 새로 mesh를 계산하면 된다. 이를 위해서 Delaunay triangulation을 사용할 수 있으며, `scipy` 라이브러리 에서 이를 제공한다.

- 현재 `transform_in_space` 함수는 메시 변형을 위해 행렬을 입력으로 받으며, 따라서 가능한 변환은 오직 선형 변환이다. 그러나 이는 단순히 비선형 변환이 구현하기 번거로워서 그렇게 했을 뿐, matrix 대신 비선형 함수 `f:R³→R³` 를 제공하고 `f((L-O) × T1) × T2 + O` 를 계산함으로써 쉽게 비선형으로 확장이 가능하다. (단 함수의 경우 broadcasting이 구현되지 않으므로 L에 벡터 대신 행렬을 공급할 경우 for 등을 통하여 column-wise로 계산해야만 한다.)
