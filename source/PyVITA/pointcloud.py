import os
import struct
from enum import Enum
import numpy as np
import laspy

SUPPORTED_EXTENSIONS = {'.las', '.laz', '.ply'}


class AxisConvention(Enum):
    """좌표계 기준축 규약.
    각 값은 (side, forward, up) 축 인덱스와 부호.
    """
    ENU = 'ENU'   # X=Side(East), Y=Forward(North), Z=Up
    EDN = 'EDN'   # X=Side(East), Y=Up,             Z=Forward(North)  (down=-Z)

    @property
    def up_index(self):
        """'위' 방향에 해당하는 축 인덱스 (0=X, 1=Y, 2=Z)."""
        return 2 if self == AxisConvention.ENU else 1

    @property
    def forward_index(self):
        """'전방' 방향에 해당하는 축 인덱스."""
        return 1 if self == AxisConvention.ENU else 2

    @property
    def side_index(self):
        return 0

    def up_vector(self):
        v = np.zeros(3)
        v[self.up_index] = 1.0
        return v

    def forward_vector(self):
        v = np.zeros(3)
        v[self.forward_index] = 1.0
        return v

    def side_vector(self):
        return np.array([1.0, 0.0, 0.0])


class PointCloud:
    last_contour_info = None

    def __init__(self):
        self.points = None          # (N, 3) float64
        self.colors = None          # (N, 3) float32, 0~1
        self.offset = None          # (3,) original LAS header offset
        self.centroid = None        # (3,)
        self.primary_axis = None    # (3,) 주축 (터널 진행 방향)
        self.secondary_axis = None  # (3,) 부축
        self.tertiary_axis = None   # (3,) 3번째 축
        self.eigenvalues = None     # (3,) PCA 고유값
        self.filepath = None
        self.convention = AxisConvention.ENU
        self.colors_rgb = None
        self.has_rgb = False
        self.obb_corners = None     # (8, 3) OBB 박스 꼭짓점

    def load(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ('.las', '.laz'):
            return self.load_las(filepath)
        elif ext == '.ply':
            return self.load_ply(filepath)
        else:
            raise ValueError(f"지원하지 않는 형식: {ext}")

    def load_las(self, filepath):
        las = laspy.read(filepath)
        self.filepath = filepath

        # LAS header offset 저장 및 적용 (지리좌표계 → 로컬 좌표)
        self.offset = np.array([
            las.header.x_offset,
            las.header.y_offset,
            las.header.z_offset
        ])

        self.points = np.column_stack([
            las.x - self.offset[0],
            las.y - self.offset[1],
            las.z - self.offset[2]
        ]).astype(np.float64)

        # 색상: RGB 있으면 사용, 없으면 높이 기반 colormap
        self._load_colors(las)

        self.centroid = np.mean(self.points, axis=0)
        self.primary_axis = None
        self.secondary_axis = None
        self.tertiary_axis = None

        return len(self.points)

    def load_ply(self, filepath):
        self.filepath = filepath
        points, colors = self._parse_ply(filepath)

        # PLY는 보통 로컬 좌표이지만, 큰 좌표값이면 centroid를 offset으로 사용
        mean_pos = np.mean(points, axis=0)
        if np.any(np.abs(mean_pos) > 100000):
            self.offset = mean_pos.copy()
            self.points = points - self.offset
        else:
            self.offset = np.zeros(3)
            self.points = points

        if colors is not None:
            self.colors_rgb = colors
            self.has_rgb = True
        else:
            self.colors_rgb = None
            self.has_rgb = False
        # 기본 색상: 높이 colormap
        self.colors = self._make_height_colors()

        self.centroid = np.mean(self.points, axis=0)
        self.primary_axis = None
        self.secondary_axis = None
        self.tertiary_axis = None

        return len(self.points)

    def export_xyz_ply(self, filepath, points):
        """Save xyz-only binary PLY in original/world coordinates."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        offset = self.offset if self.offset is not None else np.zeros(3, dtype=np.float64)
        world_points = np.ascontiguousarray(points + offset[None, :], dtype='<f8')

        with open(filepath, 'wb') as f:
            header = [
                'ply',
                'format binary_little_endian 1.0',
                'comment SAMSLab xyz-only aligned export',
                f'element vertex {len(world_points)}',
                'property double x',
                'property double y',
                'property double z',
                'end_header',
            ]
            f.write(('\n'.join(header) + '\n').encode('ascii'))
            f.write(world_points.tobytes())
        return True

    @staticmethod
    def _parse_ply(filepath):
        with open(filepath, 'rb') as f:
            # header
            line = f.readline().decode('ascii').strip()
            if line != 'ply':
                raise ValueError("유효한 PLY 파일이 아닙니다.")

            fmt = None
            vertex_count = 0
            properties = []
            in_vertex = False

            while True:
                line = f.readline().decode('ascii').strip()
                if line == 'end_header':
                    break
                parts = line.split()
                if parts[0] == 'format':
                    fmt = parts[1]
                elif parts[0] == 'element' and parts[1] == 'vertex':
                    vertex_count = int(parts[2])
                    in_vertex = True
                elif parts[0] == 'element' and parts[1] != 'vertex':
                    in_vertex = False
                elif parts[0] == 'property' and in_vertex:
                    if parts[1] == 'list':
                        # property list <count_type> <elem_type> <name> — 스킵
                        continue
                    properties.append((parts[1], parts[2]))

            # property 인덱스 매핑
            prop_names = [p[1] for p in properties]
            prop_types = [p[0] for p in properties]
            # 소문자로 매핑 시도
            prop_lower = [n.lower() for n in prop_names]

            print(f"[PLY] format: {fmt}, vertices: {vertex_count}, "
                  f"properties: {list(zip(prop_types, prop_names))}")

            def find_prop(candidates):
                for c in candidates:
                    if c in prop_lower:
                        return prop_lower.index(c)
                return -1

            ix = find_prop(['x'])
            iy = find_prop(['y'])
            iz = find_prop(['z'])
            if ix < 0 or iy < 0 or iz < 0:
                raise ValueError(
                    f"PLY에 x/y/z 좌표가 없습니다. "
                    f"발견된 속성: {prop_names}")

            ir = find_prop(['red', 'r'])
            ig = find_prop(['green', 'g'])
            ib = find_prop(['blue', 'b'])
            has_rgb = ir >= 0 and ig >= 0 and ib >= 0

            type_map = {
                'float': ('f', 4), 'float32': ('f', 4), 'double': ('d', 8), 'float64': ('d', 8),
                'uchar': ('B', 1), 'uint8': ('B', 1), 'char': ('b', 1), 'int8': ('b', 1),
                'ushort': ('H', 2), 'uint16': ('H', 2), 'short': ('h', 2), 'int16': ('h', 2),
                'uint': ('I', 4), 'uint32': ('I', 4), 'int': ('i', 4), 'int32': ('i', 4),
            }

            if fmt == 'ascii':
                points = np.zeros((vertex_count, 3), dtype=np.float64)
                colors = np.zeros((vertex_count, 3), dtype=np.float32) if has_rgb else None
                for i in range(vertex_count):
                    vals = f.readline().decode('ascii').split()
                    points[i] = [float(vals[ix]), float(vals[iy]), float(vals[iz])]
                    if has_rgb:
                        colors[i] = [float(vals[ir]), float(vals[ig]), float(vals[ib])]
                if has_rgb and colors.max() > 1.0:
                    colors /= 255.0
            else:
                # binary — numpy로 고속 읽기
                endian = '<' if 'little' in fmt else '>'

                # numpy dtype 매핑
                np_type_map = {
                    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
                    'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
                    'ushort': 'u2', 'uint16': 'u2', 'short': 'i2', 'int16': 'i2',
                    'uint': 'u4', 'uint32': 'u4', 'int': 'i4', 'int32': 'i4',
                }

                dt = np.dtype([(f'f{j}', endian + np_type_map[t]) for j, t in enumerate(prop_types)])
                raw = f.read(dt.itemsize * vertex_count)
                data = np.frombuffer(raw, dtype=dt, count=vertex_count)

                points = np.column_stack([
                    data[f'f{ix}'].astype(np.float64),
                    data[f'f{iy}'].astype(np.float64),
                    data[f'f{iz}'].astype(np.float64),
                ])

                if has_rgb:
                    r = data[f'f{ir}'].astype(np.float32)
                    g = data[f'f{ig}'].astype(np.float32)
                    b = data[f'f{ib}'].astype(np.float32)
                    colors = np.column_stack([r, g, b])
                    if colors.max() > 1.0:
                        colors /= 255.0
                else:
                    colors = None

        return points, colors

    def _load_colors(self, las):
        try:
            r = np.array(las.red, dtype=np.float32)
            g = np.array(las.green, dtype=np.float32)
            b = np.array(las.blue, dtype=np.float32)
            max_val = max(r.max(), g.max(), b.max(), 1.0)
            if max_val > 255:
                max_val = 65535.0
            else:
                max_val = 255.0
            self.colors_rgb = np.column_stack([r, g, b]) / max_val
            self.has_rgb = True
        except Exception:
            self.colors_rgb = None
            self.has_rgb = False
        # 기본 색상: 높이 colormap
        self.colors = self._make_height_colors()

    def apply_color_mode(self, mode):
        """색상 모드 적용. mode: 'height' 또는 'rgb'"""
        if mode == 'rgb' and self.has_rgb:
            self.colors = self.colors_rgb.copy()
        else:
            self.colors = self._make_height_colors()

    def _make_height_colors(self):
        """convention의 up축 기준 높이 colormap (blue→red)."""
        h = self.points[:, self.convention.up_index]
        h_min, h_max = h.min(), h.max()
        h_range = h_max - h_min if h_max > h_min else 1.0
        t = (h - h_min) / h_range  # 0=최저, 1=최고
        return self._blue_to_red_colormap(t)

    @staticmethod
    def _blue_to_red_colormap(t):
        """높이 정규값(0~1)을 blue→red colormap으로 변환.
        0.0 = blue(0,0,1), 0.5 = green(0,1,0), 1.0 = red(1,0,0)"""
        r = np.clip(2.0 * t - 1.0, 0, 1)
        g = np.clip(1.0 - np.abs(2.0 * t - 1.0), 0, 1)
        b = np.clip(1.0 - 2.0 * t, 0, 1)
        return np.column_stack([r, g, b]).astype(np.float32)

    def find_axes(self, method='PCA'):
        """수평면에서 주축/부축 계산.
        method: 'PCA' 또는 'OBB'"""
        if self.points is None:
            return None
        if method == 'OBB':
            return self._find_axes_obb()
        return self._find_axes_pca()

    def _get_horiz_info(self):
        """convention 기반 수평면 인덱스와 2D 좌표 반환."""
        conv = self.convention
        up_idx = conv.up_index
        horiz = [i for i in range(3) if i != up_idx]
        pts2d = self.points[:, horiz]
        return conv, up_idx, horiz, pts2d

    def _set_axes_from_2d(self, horiz, long_2d, short_2d, ev0, ev1, up_extent):
        """2D 축 벡터를 3D로 확장하여 주축/부축/3차축 설정."""
        def to_3d(vec2d):
            v = np.zeros(3)
            v[horiz[0]] = vec2d[0]
            v[horiz[1]] = vec2d[1]
            return v
        self.primary_axis = to_3d(long_2d)
        self.secondary_axis = to_3d(short_2d)
        self.tertiary_axis = self.convention.up_vector()
        self.eigenvalues = np.array([ev0, ev1, up_extent])

    def _find_axes_pca(self):
        """PCA 방식: 최대 분산 방향."""
        conv, up_idx, horiz, pts2d = self._get_horiz_info()
        centered = pts2d - pts2d.mean(axis=0)
        cov2d = np.cov(centered, rowvar=False)

        eigenvalues_2d, eigenvectors_2d = np.linalg.eigh(cov2d)
        idx = np.argsort(eigenvalues_2d)[::-1]
        eigenvalues_2d = eigenvalues_2d[idx]
        eigenvectors_2d = eigenvectors_2d[:, idx]

        up_var = np.var(self.points[:, up_idx])
        self._set_axes_from_2d(
            horiz,
            eigenvectors_2d[:, 0], eigenvectors_2d[:, 1],
            eigenvalues_2d[0], eigenvalues_2d[1], up_var
        )

        # PCA 축 기준 OBB 박스 계산
        self._compute_obb_from_axes(horiz, up_idx)

        print(f"[PCA] Primary: {self.primary_axis}, EV ratio: "
              f"{eigenvalues_2d[0]/eigenvalues_2d.sum():.1%} / "
              f"{eigenvalues_2d[1]/eigenvalues_2d.sum():.1%}")
        return self.primary_axis, self.secondary_axis, self.tertiary_axis

    def _compute_obb_from_axes(self, horiz, up_idx):
        """현재 primary/secondary 축 기준으로 OBB 8꼭짓점 계산."""
        # 2D 축 벡터 추출
        ax_primary = np.array([self.primary_axis[horiz[0]], self.primary_axis[horiz[1]]])
        ax_secondary = np.array([self.secondary_axis[horiz[0]], self.secondary_axis[horiz[1]]])

        pts2d = self.points[:, horiz]
        # 축 방향으로 투영
        proj_p = pts2d @ ax_primary
        proj_s = pts2d @ ax_secondary
        p_min, p_max = proj_p.min(), proj_p.max()
        s_min, s_max = proj_s.min(), proj_s.max()

        up_min = self.points[:, up_idx].min()
        up_max = self.points[:, up_idx].max()

        # 4개 2D 코너 (primary x secondary 조합)
        corners_2d = np.array([
            ax_primary * p_min + ax_secondary * s_min,
            ax_primary * p_max + ax_secondary * s_min,
            ax_primary * p_max + ax_secondary * s_max,
            ax_primary * p_min + ax_secondary * s_max,
        ])

        self.obb_corners = np.zeros((8, 3))
        for i in range(4):
            self.obb_corners[i, horiz[0]] = corners_2d[i, 0]
            self.obb_corners[i, horiz[1]] = corners_2d[i, 1]
            self.obb_corners[i, up_idx] = up_min
            self.obb_corners[i + 4, horiz[0]] = corners_2d[i, 0]
            self.obb_corners[i + 4, horiz[1]] = corners_2d[i, 1]
            self.obb_corners[i + 4, up_idx] = up_max

    def _find_axes_obb(self):
        """OBB 방식: 최소 면적 바운딩 박스의 장축/단축.
        수평면에서 -90~+90도, 1도 간격으로 회전하며 최소 면적 OBB 탐색."""
        conv, up_idx, horiz, pts2d = self._get_horiz_info()

        best_angle = 0.0
        best_area = np.inf
        best_w = 0.0
        best_h = 0.0

        for deg in range(-90, 91):
            rad = np.radians(deg)
            c, s = np.cos(rad), np.sin(rad)
            # 2D 회전
            rx = c * pts2d[:, 0] + s * pts2d[:, 1]
            ry = -s * pts2d[:, 0] + c * pts2d[:, 1]
            w = rx.max() - rx.min()
            h = ry.max() - ry.min()
            area = w * h
            if area < best_area:
                best_area = area
                best_angle = deg
                best_w, best_h = w, h

        rad = np.radians(best_angle)
        c, s = np.cos(rad), np.sin(rad)
        # 회전 좌표계의 X축 = (c, s), Y축 = (-s, c)
        axis_x = np.array([c, s])
        axis_y = np.array([-s, c])

        # 장축 = 긴 쪽, 단축 = 짧은 쪽
        if best_h >= best_w:
            long_2d, short_2d = axis_y, axis_x
            long_ext, short_ext = best_h, best_w
        else:
            long_2d, short_2d = axis_x, axis_y
            long_ext, short_ext = best_w, best_h

        up_extent = self.points[:, up_idx].max() - self.points[:, up_idx].min()
        self._set_axes_from_2d(horiz, long_2d, short_2d, long_ext, short_ext, up_extent)
        self._compute_obb_from_axes(horiz, up_idx)

        print(f"[OBB] Angle: {best_angle}° | "
              f"Extents: {long_ext:.2f} x {short_ext:.2f} | "
              f"Primary: {self.primary_axis}")
        return self.primary_axis, self.secondary_axis, self.tertiary_axis

    def _build_rotation_matrix(self):
        """주축/부축/3차축으로부터 회전 행렬 구성."""
        conv = self.convention
        R = np.zeros((3, 3))
        R[conv.side_index] = self.secondary_axis
        R[conv.forward_index] = self.primary_axis
        R[conv.up_index] = self.tertiary_axis
        if np.linalg.det(R) < 0:
            R[conv.up_index, :] = -R[conv.up_index, :]
        return R

    def make_aligned(self):
        """원본을 수정하지 않고 정렬된 좌표와 색상을 반환.
        정렬 데이터의 중심을 원본 centroid와 동일하게 유지.
        Returns: (aligned_points, colors) or None"""
        if self.primary_axis is None:
            return None

        R = self._build_rotation_matrix()
        centered = self.points - self.centroid
        aligned_points = (R @ centered.T).T + self.centroid
        return aligned_points, self.colors.copy()

    def align(self):
        """convention에 따라 주축→forward, 부축→side, 3차축→up 정렬 (원본 수정)."""
        if self.primary_axis is None:
            return False

        R = self._build_rotation_matrix()
        centered = self.points - self.centroid
        self.points = (R @ centered.T).T

        # 정렬 후 범위 확인
        pmin = self.points.min(axis=0)
        pmax = self.points.max(axis=0)
        ranges = pmax - pmin
        print(f"[Align] Ranges after: X={ranges[0]:.2f}, Y={ranges[1]:.2f}, Z={ranges[2]:.2f}")
        print(f"[Align] Forward({conv.forward_index})={ranges[conv.forward_index]:.2f} should be largest horizontal")

        self.centroid = np.mean(self.points, axis=0)

        # 축 정보를 convention 기준으로 갱신
        self.primary_axis = conv.forward_vector()
        self.secondary_axis = conv.side_vector()
        self.tertiary_axis = conv.up_vector()

        return True

    def get_bounds(self):
        """포인트 클라우드의 AABB 반환."""
        if self.points is None:
            return None
        return self.points.min(axis=0), self.points.max(axis=0)

    @staticmethod
    def extract_contour(points_2d, method='radial', **kwargs):
        """2D 포인트의 외곽 추출.
        Args:
            points_2d: (N, 2) array
            method: 'radial', 'radial_percentile', 'grid_ms', 'radial_midline', 'convex', 'concave', 'alpha', 'circle'
            kwargs:
                num_bins: radial 방식 bin 수 (기본 360)
                percentile: radial percentile 방식 백분위 (기본 98)
                grid_size: grid_ms 방식 셀 크기 (기본 0.05)
                outer_percentile: radial midline 방식 바깥 백분위 (기본 90)
                alpha: alpha shape 반경 (기본 자동)
                concavity: concave hull 오목 정도 (기본 2.0)
        Returns:
            contour: (M, 2) array - 외곽 폴리라인 점 목록
        """
        if len(points_2d) < 3:
            PointCloud.last_contour_info = None
            return points_2d.copy()

        PointCloud.last_contour_info = None
        if method == 'convex':
            return PointCloud._contour_convex(points_2d)
        elif method == 'concave':
            return PointCloud._contour_concave(points_2d, kwargs.get('concavity', 2.0))
        elif method == 'circle':
            return PointCloud._contour_top_circle(points_2d, kwargs.get('num_pts', 120))
        elif method == 'alpha':
            return PointCloud._contour_alpha(points_2d, kwargs.get('alpha', 0.0))
        elif method == 'radial_percentile':
            return PointCloud._contour_radial_percentile(
                points_2d,
                kwargs.get('num_bins', 360),
                kwargs.get('percentile', 98.0),
            )
        elif method == 'grid_ms':
            return PointCloud._contour_grid_ms(points_2d, kwargs.get('grid_size', 0.05))
        elif method == 'five_circle':
            from five_circle_fit import fit_five_circle_contour
            fit_points = points_2d
            if kwargs.get('preprocess_radial', False):
                radial_points = PointCloud._contour_radial(
                    points_2d,
                    kwargs.get('preprocess_radial_bins', 360),
                    kwargs.get('preprocess_radial_min_radius', 1.0),
                    None,
                    kwargs.get('preprocess_radial_midline', True),
                    kwargs.get('preprocess_radial_outer_percentile', 90.0),
                )
                if len(radial_points) >= 10:
                    fit_points = radial_points
            contour, info = fit_five_circle_contour(
                fit_points,
                n_per_arc=kwargs.get('n_per_arc', 160),
                n_circle=kwargs.get('n_circle', 240),
                debug_path=kwargs.get('debug_path'),
                debug_meta=kwargs.get('debug_meta'),
            )
            if kwargs.get('preprocess_radial', False):
                info = dict(info or {})
                info['preprocess_radial'] = {
                    'enabled': True,
                    'input_count': int(len(points_2d)),
                    'fit_count': int(len(fit_points)),
                    'bins': int(kwargs.get('preprocess_radial_bins', 360)),
                    'min_radius': float(kwargs.get('preprocess_radial_min_radius', 1.0)),
                    'midline': bool(kwargs.get('preprocess_radial_midline', True)),
                    'outer_percentile': float(kwargs.get('preprocess_radial_outer_percentile', 90.0)),
                }
            PointCloud.last_contour_info = info
            return contour
        elif method == 'radial_midline':
            return PointCloud._contour_radial_midline(
                points_2d,
                kwargs.get('num_bins', 360),
                kwargs.get('outer_percentile', 90.0),
            )
        else:  # radial
            return PointCloud._contour_radial(
                points_2d,
                kwargs.get('num_bins', 360),
                kwargs.get('min_radius', 0.0),
                kwargs.get('percentile', None),
                kwargs.get('midline', False),
                kwargs.get('outer_percentile', 90.0),
            )

    @staticmethod
    def _contour_radial(points_2d, num_bins=360, min_radius=0.0,
                        percentile=None, midline=False, outer_percentile=90.0):
        """각도 bin 최원점 방식."""
        lo = np.percentile(points_2d, 2.0, axis=0)
        hi = np.percentile(points_2d, 98.0, axis=0)
        center = (lo + hi) * 0.5
        rel = points_2d - center
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        dists = np.sqrt(rel[:, 0]**2 + rel[:, 1]**2)
        min_radius = max(float(min_radius), 0.0)
        if min_radius > 0.0:
            candidate_mask = dists >= min_radius
            if np.count_nonzero(candidate_mask) < 3:
                return points_2d[:0]
            points_2d = points_2d[candidate_mask]
            angles = angles[candidate_mask]
            dists = dists[candidate_mask]
        bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
        bin_idx = np.digitize(angles, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)
        if midline:
            outer_percentile = float(np.clip(outer_percentile, 55.0, 99.5))
            inner_percentile = 100.0 - outer_percentile
        elif percentile is not None:
            percentile = float(np.clip(percentile, 50.0, 100.0))

        contour = []
        for b in range(num_bins):
            mask = bin_idx == b
            if not np.any(mask):
                continue
            pt_indices = np.where(mask)[0]
            d_bin = dists[mask]
            if midline:
                r_in = PointCloud._fast_percentile(d_bin, inner_percentile)
                r_out = PointCloud._fast_percentile(d_bin, outer_percentile)
                r_mid = 0.5 * (r_in + r_out)
                theta = 0.5 * (bin_edges[b] + bin_edges[b + 1])
                contour.append(center + r_mid * np.array([np.cos(theta), np.sin(theta)]))
                continue
            if percentile is None:
                local_idx = np.argmax(d_bin)
            else:
                k = PointCloud._percentile_index(len(d_bin), percentile)
                local_idx = np.argpartition(d_bin, k)[k]
            contour.append(points_2d[pt_indices[local_idx]])

        return np.array(contour) if len(contour) >= 3 else points_2d[:0]

    @staticmethod
    def _contour_radial_percentile(points_2d, num_bins=360, percentile=98.0):
        """각도 bin별 상위 백분위 반경을 사용하는 radial 외곽."""
        radial = PointCloud._prepare_radial_bins(points_2d, num_bins)
        center, bin_edges, dists, groups = radial
        percentile = float(np.clip(percentile, 50.0, 100.0))
        contour = []

        for _bin_no, idx in groups:
            d_bin = dists[idx]
            if len(d_bin) == 0:
                continue

            k = PointCloud._percentile_index(len(d_bin), percentile)
            local_idx = np.argpartition(d_bin, k)[k]
            contour.append(points_2d[idx[local_idx]])

        return np.array(contour) if len(contour) >= 3 else points_2d[:0]

    @staticmethod
    def _contour_grid_ms(points_2d, grid_size=0.05):
        """격자화 + morphology 후 contour loop를 복원하는 방식."""
        from scipy import ndimage

        if len(points_2d) < 3:
            return points_2d.copy()

        grid_size, min_pt, _counts, mask = PointCloud._rasterize_points_to_grid(points_2d, grid_size)
        structure = np.ones((3, 3), dtype=bool)
        mask = ndimage.binary_dilation(mask, structure=structure, iterations=1)
        mask = ndimage.binary_closing(mask, structure=structure, iterations=2)
        mask = ndimage.binary_fill_holes(mask)
        mask = ndimage.binary_opening(mask, structure=structure, iterations=1)

        loops = PointCloud._mask_to_loops(mask)
        if not loops:
            return PointCloud._contour_radial_percentile(points_2d)

        contour = max(loops, key=PointCloud._polyline_area_abs)
        contour = min_pt + contour * grid_size
        contour = PointCloud._simplify_collinear_loop(contour, eps=grid_size * 1e-3)
        return contour if len(contour) >= 3 else PointCloud._contour_radial_percentile(points_2d)

    @staticmethod
    def _contour_radial_midline(points_2d, num_bins=360, outer_percentile=90.0):
        """각도 bin별 안/밖 percentile의 중간 반경으로 midline 생성."""
        radial = PointCloud._prepare_radial_bins(points_2d, num_bins)
        center, bin_edges, dists, groups = radial
        outer_percentile = float(np.clip(outer_percentile, 55.0, 99.5))
        inner_percentile = 100.0 - outer_percentile

        contour = []
        for bin_no, idx in groups:
            d_bin = dists[idx]
            if len(d_bin) == 0:
                continue

            r_in = PointCloud._fast_percentile(d_bin, inner_percentile)
            r_out = PointCloud._fast_percentile(d_bin, outer_percentile)
            r_mid = 0.5 * (r_in + r_out)
            theta = 0.5 * (bin_edges[bin_no] + bin_edges[bin_no + 1])
            contour.append(center + r_mid * np.array([np.cos(theta), np.sin(theta)]))

        return np.array(contour) if len(contour) >= 3 else points_2d[:0]

    @staticmethod
    def _rasterize_points_to_grid(points_2d, grid_size=0.05):
        """2D 포인트를 grid로 rasterize하고 occupancy mask를 반환."""
        grid_size = max(float(grid_size), 1e-3)
        margin = grid_size * 2.0
        min_pt = points_2d.min(axis=0) - margin
        max_pt = points_2d.max(axis=0) + margin
        span = np.maximum(max_pt - min_pt, grid_size * 4.0)

        nx = int(np.ceil(span[0] / grid_size)) + 1
        ny = int(np.ceil(span[1] / grid_size)) + 1
        counts = np.zeros((ny, nx), dtype=np.int32)

        ij = np.floor((points_2d - min_pt) / grid_size).astype(np.int32)
        ij[:, 0] = np.clip(ij[:, 0], 0, nx - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, ny - 1)
        np.add.at(counts, (ij[:, 1], ij[:, 0]), 1)
        mask = counts > 0
        return grid_size, min_pt, counts, mask

    @staticmethod
    def _prepare_radial_bins(points_2d, num_bins=360):
        """Radial 계열에서 공통으로 쓰는 bin group을 한 번만 구성."""
        center = np.median(points_2d, axis=0)
        rel = points_2d - center
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        dists = np.sqrt(rel[:, 0]**2 + rel[:, 1]**2)

        num_bins = int(max(num_bins, 12))
        bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
        bin_idx = np.digitize(angles, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)

        order = np.argsort(bin_idx, kind='stable')
        sorted_bins = bin_idx[order]
        groups = []
        if len(order) > 0:
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_bins)) + 1]
            ends = np.r_[starts[1:], len(order)]
            groups = [(int(sorted_bins[s]), order[s:e]) for s, e in zip(starts, ends)]

        return center, bin_edges, dists, groups

    @staticmethod
    def _percentile_index(count, percentile):
        if count <= 1:
            return 0
        percentile = float(np.clip(percentile, 0.0, 100.0))
        return int(round((percentile / 100.0) * (count - 1)))

    @staticmethod
    def _fast_percentile(values, percentile):
        if len(values) == 0:
            return 0.0
        k = PointCloud._percentile_index(len(values), percentile)
        return float(np.partition(values, k)[k])

    @staticmethod
    def _mask_to_loops(mask):
        """True mask의 바깥 경계 loop를 lattice 좌표로 반환."""
        ny, nx = mask.shape
        segments = []

        for iy in range(ny):
            for ix in range(nx):
                if not mask[iy, ix]:
                    continue
                if iy == 0 or not mask[iy - 1, ix]:
                    segments.append(((ix, iy), (ix + 1, iy)))
                if ix == nx - 1 or not mask[iy, ix + 1]:
                    segments.append(((ix + 1, iy), (ix + 1, iy + 1)))
                if iy == ny - 1 or not mask[iy + 1, ix]:
                    segments.append(((ix + 1, iy + 1), (ix, iy + 1)))
                if ix == 0 or not mask[iy, ix - 1]:
                    segments.append(((ix, iy + 1), (ix, iy)))

        if not segments:
            return []

        outgoing = {}
        for idx, (p0, p1) in enumerate(segments):
            outgoing.setdefault(p0, []).append((idx, p1))

        used = np.zeros(len(segments), dtype=bool)
        loops = []

        for idx, (start, end) in enumerate(segments):
            if used[idx]:
                continue

            used[idx] = True
            loop = [start]
            current = end

            while current != start:
                loop.append(current)
                candidates = outgoing.get(current, [])
                next_idx = -1
                next_point = None
                for cand_idx, cand_point in candidates:
                    if not used[cand_idx]:
                        next_idx = cand_idx
                        next_point = cand_point
                        break
                if next_idx < 0:
                    loop = []
                    break
                used[next_idx] = True
                current = next_point

            if len(loop) >= 3:
                loops.append(np.asarray(loop, dtype=np.float64))

        return loops

    @staticmethod
    def _polyline_area_abs(points_2d):
        if len(points_2d) < 3:
            return 0.0
        x = points_2d[:, 0]
        y = points_2d[:, 1]
        return abs(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    @staticmethod
    def _simplify_collinear_loop(points_2d, eps=1e-9):
        if len(points_2d) < 4:
            return points_2d

        simplified = []
        n = len(points_2d)
        for i in range(n):
            prev_pt = points_2d[i - 1]
            curr_pt = points_2d[i]
            next_pt = points_2d[(i + 1) % n]
            v1 = curr_pt - prev_pt
            v2 = next_pt - curr_pt
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) <= eps and np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                continue
            simplified.append(curr_pt)

        return np.asarray(simplified, dtype=np.float64)

    @staticmethod
    def _contour_convex(points_2d):
        """Convex Hull (scipy)."""
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points_2d)
        return points_2d[hull.vertices]

    @staticmethod
    def _contour_concave(points_2d, concavity=2.0):
        """Delaunay 기반 concave hull.

        concavity는 유지된 API 이름이지만 UI에서는 Max Edge(m)로 사용한다.
        Max Edge보다 긴 변을 가진 삼각형을 제거하고 남은 boundary loop를 반환한다.
        """
        points = np.unique(np.asarray(points_2d, dtype=np.float64), axis=0)
        if len(points) < 4:
            return points.copy()

        from scipy.spatial import Delaunay, QhullError

        max_edge = float(concavity)
        if max_edge <= 0.0:
            max_edge = PointCloud._estimate_concave_max_edge(points)

        try:
            tri = Delaunay(points)
        except QhullError:
            return PointCloud._contour_convex(points)

        simplices = tri.simplices
        tri_pts = points[simplices]
        e01 = np.linalg.norm(tri_pts[:, 0] - tri_pts[:, 1], axis=1)
        e12 = np.linalg.norm(tri_pts[:, 1] - tri_pts[:, 2], axis=1)
        e20 = np.linalg.norm(tri_pts[:, 2] - tri_pts[:, 0], axis=1)
        keep = np.maximum.reduce([e01, e12, e20]) <= max_edge

        if not np.any(keep):
            return points[:0]

        boundary_edges = PointCloud._boundary_edges_from_triangles(simplices[keep])
        loops = PointCloud._boundary_loops_from_edges(boundary_edges)
        if not loops:
            return points[:0]

        loop = max(loops, key=lambda idx: PointCloud._polyline_area_abs(points[idx]))
        contour = points[loop]
        if len(contour) < 3 or PointCloud._polyline_area_abs(contour) <= 1e-12:
            return points[:0]
        if PointCloud._polygon_signed_area(contour) < 0:
            contour = contour[::-1]
        return contour

    @staticmethod
    def _estimate_concave_max_edge(points):
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        dists, _ = tree.query(points, k=min(4, len(points)))
        if dists.ndim == 1 or dists.shape[1] < 2:
            return 1.0
        nn = dists[:, 1:]
        return float(np.percentile(nn[np.isfinite(nn)], 90.0) * 3.0)

    @staticmethod
    def _boundary_edges_from_triangles(triangles):
        edges = np.vstack([
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ])
        edges = np.sort(edges, axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        return unique_edges[counts == 1]

    @staticmethod
    def _boundary_loops_from_edges(edges):
        if len(edges) == 0:
            return []

        adjacency = {}
        unused = set()
        for a, b in edges:
            a = int(a)
            b = int(b)
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
            unused.add(tuple(sorted((a, b))))

        loops = []
        while unused:
            start, current = unused.pop()
            loop = [start, current]

            while True:
                candidates = []
                for nxt in adjacency.get(current, []):
                    edge = tuple(sorted((current, int(nxt))))
                    if edge in unused:
                        candidates.append(int(nxt))

                if not candidates:
                    break

                if start in candidates and len(loop) > 2:
                    nxt = start
                else:
                    non_start = [idx for idx in candidates if idx != start]
                    nxt = non_start[0] if non_start else candidates[0]

                unused.remove(tuple(sorted((current, nxt))))
                if nxt == start:
                    if len(loop) >= 3:
                        loops.append(np.asarray(loop, dtype=np.int64))
                    break

                loop.append(nxt)
                current = nxt

        return loops

    @staticmethod
    def _polygon_signed_area(poly):
        x = poly[:, 0]
        y = poly[:, 1]
        return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    @staticmethod
    def _contour_alpha(points_2d, alpha=0.0):
        """Alpha Shape. alpha=0이면 자동 계산."""
        import alphashape
        from shapely.geometry import Polygon, MultiPolygon
        if alpha <= 0:
            # 자동: 평균 최근접 거리 기반
            from scipy.spatial import cKDTree
            tree = cKDTree(points_2d)
            dists, _ = tree.query(points_2d, k=2)
            avg_dist = np.mean(dists[:, 1])
            alpha = 1.0 / (avg_dist * 2.0)
        shape = alphashape.alphashape(points_2d, alpha)
        if shape is None or shape.is_empty:
            return PointCloud._contour_convex(points_2d)
        if isinstance(shape, MultiPolygon):
            shape = max(shape.geoms, key=lambda g: g.area)
        if isinstance(shape, Polygon):
            coords = np.array(shape.exterior.coords)
            return coords[:-1]
        return PointCloud._contour_convex(points_2d)

    @staticmethod
    def _contour_top_circle(points_2d, num_pts=120):
        """단면 최상위점을 반드시 지나는 최적 원(Best-fit circle through top point).
        최상위점 + 양쪽 가장자리점 3점으로 원을 결정한 뒤 폴리라인 생성.

        알고리즘:
        1. 최상위점(Y최대) 찾기
        2. 좌측/우측 최외곽점 찾기
        3. 3점 외접원 계산
        4. 원 위의 점을 폴리라인으로 반환 (하단 절삭: Y < 좌우 외곽점 Y 최소)
        """
        if len(points_2d) < 3:
            return points_2d.copy()

        # 1. 최상위점 (Y 최대)
        top_idx = np.argmax(points_2d[:, 1])
        p_top = points_2d[top_idx]

        # 2. 좌측 최외곽 (X 최소), 우측 최외곽 (X 최대)
        left_idx = np.argmin(points_2d[:, 0])
        right_idx = np.argmax(points_2d[:, 0])
        p_left = points_2d[left_idx]
        p_right = points_2d[right_idx]

        # 3. 3점 외접원 계산
        center, radius = PointCloud._circle_from_3pts(p_top, p_left, p_right)
        if center is None or radius <= 0:
            # fallback: top + centroid로 원 추정
            cx = (p_left[0] + p_right[0]) * 0.5
            cy = p_top[1] - np.abs(p_right[0] - p_left[0]) * 0.5
            center = np.array([cx, cy])
            radius = np.linalg.norm(p_top - center)

        # 4. 원 폴리라인 생성 (하단 절삭)
        y_cutoff = min(p_left[1], p_right[1])
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        circle_pts = np.column_stack([
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        ])

        # 하단 절삭: y_cutoff 이상만
        mask = circle_pts[:, 1] >= y_cutoff
        if np.sum(mask) < 3:
            return circle_pts  # 절삭 후 부족하면 전체 반환
        return circle_pts[mask]

    @staticmethod
    def _circle_from_3pts(p1, p2, p3):
        """3점 외접원 계산. Returns (center, radius) or (None, 0)."""
        ax, ay = p1
        bx, by = p2
        cx, cy = p3
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            return None, 0
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
        center = np.array([ux, uy])
        radius = np.linalg.norm(p1 - center)
        return center, radius
