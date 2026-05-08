# PySamsLab 개발 노트

터널 숏크리트 모니터링을 위한 LiDAR 포인트 클라우드 처리 테스트 프로그램.
Python(PyQt5+OpenGL) 프로토타입 → 향후 C++ 포팅 예정.

## 프로젝트 경로

```
D:\Project\삼성물산(SMC)\2024.LiDAR숏크리트모니터링\repository\SamsLab.git\
├── .vscode/launch.json          # VSCode Debug/Release 설정
└── Source/PySamsLab/
    ├── main.py                  # 진입점
    ├── mainwindow.py            # QMainWindow (툴바, 상태바, 이벤트)
    ├── viewer3d.py              # QOpenGLWidget 3D 뷰어 + EDL 셰이더
    ├── pointcloud.py            # PointCloud 데이터 모델 + AxisConvention
    └── requirements.txt         # PyQt5, PyOpenGL, laspy, numpy
```

## 의존 라이브러리

```
PyQt5>=5.15
PyOpenGL>=3.1.6
laspy>=2.4
numpy>=1.24
```

## 실행 방법

```bash
cd Source/PySamsLab
pip install -r requirements.txt
python main.py
```

VSCode: `.vscode/launch.json`에 Debug/Release 설정 포함.

---

## 현재 구현 상태

### 1. 파일 로딩 (`pointcloud.py`)

- **LAS/LAZ**: `laspy`로 읽기. header offset(x_offset, y_offset, z_offset)을 빼서 로컬 좌표 변환.
- **PLY**: ASCII / binary_little_endian / binary_big_endian 직접 파싱. 좌표값이 큰 경우(>100000) centroid를 offset으로 사용.
- **색상**: RGB 있으면 사용 (16bit→정규화 자동 판별), 없으면 convention의 up축 기준 jet colormap.
- **드래그앤드롭**: MainWindow에서 `.las`, `.laz`, `.ply` 파일 드래그앤드롭 지원.
- **지원 확장자**: `SUPPORTED_EXTENSIONS = {'.las', '.laz', '.ply'}`

### 2. 좌표계 규약 (`AxisConvention`)

`AxisConvention` enum으로 좌표계 전환:

| 규약 | Side | Forward | Up | 설명 |
|------|------|---------|----|----|
| **ENU** (기본) | X(1,0,0) | Y(0,1,0) | Z(0,0,1) | Z-up. 뷰어에서 `glRotatef(-90,1,0,0)` 변환 |
| **EDN** | X(1,0,0) | Z(0,0,1) | Y(0,1,0) | Y-up. OpenGL 기본과 동일, 변환 없음 |

프로퍼티: `up_index`, `forward_index`, `side_index`, `up_vector()`, `forward_vector()`, `side_vector()`

convention에 따라 자동 연동되는 항목:
- 뷰어 카메라 회전/패닝 축
- 높이 colormap 기준축
- PCA 수평면 (2D PCA)
- 정렬 시 축 매핑

### 3. 방향 찾기 (PCA) (`pointcloud.py: find_axes`)

- convention의 **수평면에 투영 후 2D PCA** 수행 (ENU→XY평면, EDN→XZ평면).
- up축 성분을 제거하여 주축/부축이 반드시 수평면 내에 존재.
- 3차축은 convention의 up 방향으로 고정.
- `numpy.linalg.eigh` → 내림차순 정렬.

### 4. 정렬 (`pointcloud.py: align`)

- 주축 → convention의 forward축
- 부축 → convention의 side축
- 3차축 → convention의 up축
- `det(R) < 0`이면 up축 반전하여 오른손 좌표계 보장.
- 정렬 후 축 벡터를 convention 기본 벡터로 갱신.

### 5. 3D 뷰어 (`viewer3d.py`)

**렌더링:**
- Vertex array (numpy → `glVertexPointer` / `glColorPointer`)로 GL_POINTS 렌더링.
- 축 라인은 depth test 비활성화 후 그려서 점군 위에 항상 표시.
- XYZ 축 레이블: `gluProject`로 3D→2D 변환 후 `QPainter` 오버레이.

**카메라 (터넴테이블):**
- azimuth: up축(GL Y) 기준 수평 회전
- elevation: 수평면에서 올려보는 각도 (-90~90)
- 좌클릭 드래그: 회전 (azimuth += dx, elevation += dy)
- 우클릭 드래그: 패닝 (convention의 side/up축 기준)
- 휠: 줌 (distance 조절)

**ModelView 변환 순서 (역순 적용):**
1. pan 이동 (월드 좌표)
2. 좌표계 변환 (ENU: `glRotatef(-90,1,0,0)`, EDN: 없음)
3. azimuth 회전 (`glRotatef(azimuth, 0,1,0)`)
4. elevation 회전 (`glRotatef(elevation, 1,0,0)`)
5. 카메라 후퇴 (`glTranslatef(0,0,-distance)`)

**투영 모드:**
- Perspective: `gluPerspective(45, aspect, near, far)`
- Orthographic: `glOrtho` (distance에 비례)

**클리핑 거리:**
- near = `max(distance * 0.001, 0.01)`
- far = `distance + scene_radius * 4.0`
- `_scene_radius`는 포인트 클라우드 바운딩 박스 대각선의 절반.

**뷰 프리셋:**
- Top View: azimuth=0, elevation=90
- Front View: azimuth=0, elevation=0
- Right View: azimuth=90, elevation=0
- 단축키: Numpad 7/1/3

### 6. EDL (Eye Dome Lighting) (`viewer3d.py`)

**구현 방식 (Potree 방식):**
1. **1st pass**: 커스텀 셰이더로 점군 렌더링 → RGBA32F FBO
   - RGB = 포인트 색상
   - Alpha = `log2(eye_distance)` (Potree/Cesium 공통 방식)
   - Depth renderbuffer는 Z-test 전용 (셰이더에서 읽지 않음)
2. **2nd pass**: EDL 포스트프로세스
   - `response = avg(max(0, depth_center - depth_neighbor))` (8방향)
   - `shade = exp(-response * 300.0 * strength)`
   - 배경(alpha=0)은 원래 배경색 유지

**log2 depth를 사용하는 이유 (핵심):**
- Linear depth: 50m vs 50.1m 차이가 큼 → 같은 표면의 인접 포인트도 음영 발생 (per-point artifact)
- Log2 depth: `log2(50.1) - log2(50.0) = 0.003` → 거의 0 → 같은 표면은 매끄러움
- 엣지: `log2(50) - log2(200) = 2.0` → 강한 윤곽 (depth 불연속에만 반응)

**FBO 관리:**
- RGBA32F 색상 텍스처 + GL_DEPTH_COMPONENT24 렌더버퍼
- HiDPI 대응: `devicePixelRatioF()` 기반 FBO 크기
- **주의**: QOpenGLWidget은 내부 FBO → `self.defaultFramebufferObject()` 사용 (0이 아님)

**파라미터:**
- strength: 0.1~5.0, 기본 1.0 (Potree 기본값)
- radius: 1.4 pixels
- 매직 넘버 300.0은 log2 스케일 보정 (Potree/Cesium 공통)

**투영 모드 UI:**
- `| 원근 | 정사 |` QActionGroup 라디오 버튼

---

## 중요 교훈

### QOpenGLWidget depth buffer 필수 설정
`main.py`에서 `QSurfaceFormat.setDefaultFormat()`으로 **24bit depth buffer를 명시 요청**해야 함.
설정하지 않으면 일부 Windows GPU 드라이버에서 depth buffer가 0bit로 생성되어:
- `glEnable(GL_DEPTH_TEST)` 호출해도 depth test 무동작
- 뒤의 포인트가 앞의 포인트를 덮어씀
- EDL FBO의 depth renderbuffer도 무효화
```python
fmt = QSurfaceFormat()
fmt.setDepthBufferSize(24)
QSurfaceFormat.setDefaultFormat(fmt)  # QApplication 생성 전에 호출
```

## 알려진 이슈 / TODO

- [ ] EDL multi-scale 미구현 (CloudCompare는 3단계 + bilateral filter). 현재 단일 스케일.
- [ ] 대용량 점군 성능: 현재 vertex array 방식. VBO/VAO로 개선 가능.
- [x] Slice & 단면 보기 기능 구현 완료
- [ ] 다음 단계: 숏크리트 두께 계산, 2D grid 시각화.

---

## 향후 계획

1. **단면 추출**: 정렬된 점군에서 Y(forward)축 기준으로 슬라이스 → 2D 단면 벡터 생성
2. **숏크리트 두께 계산**: 발파 후 단면 vs 숏크리트 후 점군 비교
3. **2D Grid 시각화**: 두께 분포를 2D heatmap으로 표시
4. **C++ 포팅**: 동일 구조로 Qt/OpenGL C++ 구현
