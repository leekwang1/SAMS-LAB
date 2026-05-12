import numpy as np
from PyQt5.QtWidgets import (
    QOpenGLWidget, QWidget, QVBoxLayout, QToolBar, QAction,
    QActionGroup, QSlider, QLabel
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QFont, QColor
from OpenGL.GL import *
from OpenGL.GL import shaders as gl_shaders
from OpenGL.GLU import *

from pointcloud import AxisConvention

# ── 일반 렌더링용 원형 포인트 셰이더 ──

_CIRCLE_VERT_SRC = """
#version 120
uniform float uPointSize;
void main() {
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    gl_FrontColor = gl_Color;
    gl_PointSize = uPointSize;
}
"""

_CIRCLE_FRAG_SRC = """
#version 120
void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25)
        discard;
    gl_FragColor = gl_Color;
}
"""

# ── EDL 1st pass: 포인트 렌더링 + alpha에 log2(depth) 저장 ──

_DEPTH_VERT_SRC = """
#version 120
uniform float uPointSize;
varying float vLogDepth;
void main() {
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    gl_FrontColor = gl_Color;
    gl_PointSize = uPointSize;
    float eyeZ = -(gl_ModelViewMatrix * gl_Vertex).z;
    vLogDepth = log2(max(eyeZ, 0.001));
}
"""

_DEPTH_FRAG_SRC = """
#version 120
uniform bool uCircle;
varying float vLogDepth;
void main() {
    if (uCircle) {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (dot(coord, coord) > 0.25)
            discard;
    }
    gl_FragColor = vec4(gl_Color.rgb, vLogDepth);
}
"""

# ── 2nd pass: EDL shade (CloudCompare 방식) ──

_EDL_VERT_SRC = """
#version 120
void main() {
    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_Vertex;
}
"""

_EDL_FRAG_SRC = """
#version 120
uniform sampler2D uColorTex;
uniform float uScreenWidth;
uniform float uScreenHeight;
uniform float uRadius;
uniform float uStrength;

float response(float depth, vec2 uv) {
    vec2 uvRadius = uRadius / vec2(uScreenWidth, uScreenHeight);
    float sum = 0.0;
    int count = 0;

    vec2 neighbours[8];
    neighbours[0] = vec2( 1.0,  0.0);
    neighbours[1] = vec2(-1.0,  0.0);
    neighbours[2] = vec2( 0.0,  1.0);
    neighbours[3] = vec2( 0.0, -1.0);
    neighbours[4] = vec2( 0.707,  0.707);
    neighbours[5] = vec2(-0.707,  0.707);
    neighbours[6] = vec2( 0.707, -0.707);
    neighbours[7] = vec2(-0.707, -0.707);

    for (int i = 0; i < 8; i++) {
        vec2 uvN = uv + uvRadius * neighbours[i];
        float Zn = texture2D(uColorTex, uvN).a;
        if (Zn != 0.0) {
            sum += max(0.0, depth - Zn);
            count++;
        }
    }
    return (count > 0) ? sum / float(count) : 0.0;
}

void main() {
    vec2 uv = gl_TexCoord[0].st;
    vec4 color = texture2D(uColorTex, uv);
    float depth = color.a;

    // 배경 (alpha=0): 원래 색상 유지
    if (depth == 0.0) {
        discard;
    }

    float res = response(depth, uv);
    float shade = exp(-res * 300.0 * uStrength);
    gl_FragColor = vec4(color.rgb * shade, 1.0);
}
"""


class _Layer:
    """뷰어 내부 렌더링 레이어."""
    __slots__ = ('name', 'vertices', 'colors', 'count', 'visible')

    def __init__(self, name, points, colors):
        self.name = name
        self.vertices = np.ascontiguousarray(points, dtype=np.float32)
        self.colors = np.ascontiguousarray(colors, dtype=np.float32)
        self.count = len(points)
        self.visible = True


class Viewer3DWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.point_size = 2.0
        self.point_smooth = False   # True=원형, False=사각형
        self.blending = False       # True=블렌딩 on

        self._convention = AxisConvention.ENU

        self._azimuth = -45.0
        self._elevation = 30.0
        self._distance = 50.0
        self._pan_screen = np.zeros(2)
        self._center = np.zeros(3)
        self._scene_radius = 50.0

        self._perspective = True
        self._clip_near = 0.01
        self._clip_far = 10000.0

        # 레이어
        self._layers = []  # List[_Layer]

        # Slice 평면
        self._slice_planes = []
        self._active_slice = None

        # 폴리라인 + vertex
        self._polyline = None
        self._guide_polylines = []
        self._guide_points = []
        self._polyline_vtx_visible = True
        self._polyline_vtx_size = 5.0

        # 마커 (최대거리 점 표시)
        self._marker_pos = None
        self._marker_size = 0.05

        # 거리 측정
        self._measure_active = False
        self._measure_points = []

        # 행렬 캐시 (스크린→월드 변환용)
        self._last_modelview = None
        self._last_projection = None
        self._last_viewport = None

        # segment 강조
        self._highlight_seg = None  # (2, 3) float32 or None

        # 메시 (외곽 면)
        self._mesh_vertices = None   # (M, 3) float32
        self._mesh_normals = None    # (M, 3) float32
        self._mesh_count = 0
        self._mesh_visible = True

        # OBB 박스: {layer_name: (corners_8x3, visible)}
        self._obb_dict = {}  # {"원본": array, "정렬": array, ...}
        self._obb_visible = True

        # 셰이더
        self._circle_shader = None

        # EDL
        self._edl_enabled = False
        self._edl_strength = 1.0
        self._edl_radius = 1.4
        self._edl_fbo = None
        self._edl_color_tex = None
        self._edl_depth_rb = None
        self._edl_depth_shader = None
        self._edl_shade_shader = None
        self._edl_fbo_size = (0, 0)

        self._last_pos = QPoint()
        self._axes = []
        self._show_world_axes = True

        self.setMinimumSize(400, 300)

    # ── Properties ──

    @property
    def convention(self):
        return self._convention

    @convention.setter
    def convention(self, conv):
        self._convention = conv
        self.update()

    @property
    def perspective(self):
        return self._perspective

    @perspective.setter
    def perspective(self, val):
        self._perspective = val
        self.update()

    @property
    def edl_enabled(self):
        return self._edl_enabled

    @edl_enabled.setter
    def edl_enabled(self, val):
        self._edl_enabled = val
        self.update()

    # ── 레이어 관리 ──

    def add_layer(self, name, points, colors):
        """레이어 추가 (동일 name이면 교체)."""
        self.remove_layer(name)
        self._layers.append(_Layer(name, points, colors))
        self._fit_camera()
        self.update()

    def remove_layer(self, name):
        """이름으로 레이어 제거."""
        self._layers = [l for l in self._layers if l.name != name]

    def set_layer_visible(self, name, visible):
        for l in self._layers:
            if l.name == name:
                l.visible = visible
                break
        self._fit_camera()
        self.update()

    def clear_layers(self):
        self._layers.clear()
        self._axes = []
        self._slice_planes = []
        self._active_slice = None
        self._polyline = None
        self._guide_polylines = []
        self._guide_points = []
        self._obb_dict.clear()
        self._mesh_vertices = None
        self._mesh_normals = None
        self._mesh_count = 0
        self._marker_pos = None
        self.update()

    def set_point_cloud(self, points, colors, centroid):
        """하위 호환: '원본' 레이어로 추가."""
        self.clear_layers()
        self.add_layer("원본", points, colors)

    def _fit_camera(self):
        """visible 레이어 전체 bounds로 카메라 맞춤."""
        all_pts = [l.vertices for l in self._layers if l.visible and l.count > 0]
        if not all_pts:
            return
        merged = np.vstack(all_pts)
        pmin = merged.min(axis=0)
        pmax = merged.max(axis=0)
        center = (pmin + pmax) * 0.5
        bounds_range = pmax - pmin
        self._scene_radius = float(np.linalg.norm(bounds_range)) * 0.5
        self._distance = self._scene_radius * 1.6
        self._center = -center
        self._pan_screen = np.zeros(2)

    def set_axes(self, centroid, axes_list):
        self._axes = [(centroid, d, c, l) for d, c, l in axes_list]
        self.update()

    def clear_axes(self):
        self._axes = []
        self.update()

    # ── Slice 평면 ──

    def set_slice_planes(self, y_positions, x_range, z_range):
        """slice 평면 목록 설정. x_range=(xmin,xmax), z_range=(zmin,zmax)."""
        xmin, xmax = x_range
        zmin, zmax = z_range
        self._slice_planes = [(y, xmin, xmax, zmin, zmax) for y in y_positions]
        self.update()

    def set_active_slice(self, y_pos, thickness):
        """활성 slice 하이라이트."""
        self._active_slice = (y_pos, thickness)
        self.update()

    def clear_slices(self):
        self._slice_planes = []
        self._active_slice = None
        self.update()

    # ── 폴리라인 ──

    def set_polyline(self, points_3d):
        """외곽 폴리라인 설정. points_3d: (M, 3) array."""
        self._polyline = np.ascontiguousarray(points_3d, dtype=np.float32)
        self.update()

    def set_guide_polylines(self, polylines):
        """보조 폴리라인 설정. polylines: [(points_3d, color_rgb), ...]."""
        self._guide_polylines = [
            (np.ascontiguousarray(points, dtype=np.float32), tuple(color))
            for points, color in polylines
            if points is not None and len(points) >= 2
        ]
        self.update()

    def set_guide_points(self, points):
        guide_points = []
        for item in points:
            if len(item) < 3:
                continue
            point, color, size = item[:3]
            label = item[3] if len(item) >= 4 else ""
            if point is not None and len(point) == 3:
                guide_points.append((np.asarray(point, dtype=np.float32), tuple(color), float(size), str(label)))
        self._guide_points = guide_points
        self.update()

    def clear_guide_polylines(self):
        self._guide_polylines = []
        self._guide_points = []
        self.update()

    def clear_polyline(self):
        self._polyline = None
        self._guide_polylines = []
        self._guide_points = []
        self.update()

    def set_marker(self, pos_3d, size=None):
        """최대거리 점 마커 설정."""
        self._marker_pos = np.array(pos_3d, dtype=np.float32)
        if size is not None:
            self._marker_size = size
        self.update()

    def clear_marker(self):
        self._marker_pos = None
        self.update()

    # ── 거리 측정 ──

    def set_measure_active(self, active):
        self._measure_active = active
        if not active:
            self._measure_points = []
        self.setCursor(Qt.CrossCursor if active else Qt.ArrowCursor)
        self.update()

    def clear_measure(self):
        self._measure_points = []
        self.update()

    def _screen_to_world(self, sx, sy):
        """스크린 좌표 → 월드 좌표 (저장된 행렬 사용, depth=0.5)."""
        self.makeCurrent()
        dpr = self.devicePixelRatioF()
        fx = sx * dpr
        fy = (self.height() - sy) * dpr

        # paintGL에서 저장한 행렬 사용
        mv = self._last_modelview
        proj = self._last_projection
        vp = self._last_viewport

        if mv is None or proj is None or vp is None:
            return None

        # depth=0.5 (중간 깊이) 에서 unproject
        try:
            wx, wy, wz = gluUnProject(fx, fy, 0.5, mv, proj, vp)
            return np.array([wx, wy, wz])
        except Exception:
            return None

    def set_mesh(self, vertices, normals):
        """삼각형 메시 설정. vertices: (N*3, 3), normals: (N*3, 3)."""
        self._mesh_vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self._mesh_normals = np.ascontiguousarray(normals, dtype=np.float32)
        self._mesh_count = len(vertices)
        self.update()

    def clear_mesh(self):
        self._mesh_vertices = None
        self._mesh_normals = None
        self._mesh_count = 0
        self.update()

    # ── OBB 박스 ──

    def set_obb(self, name, corners_8x3):
        """레이어별 OBB 설정."""
        self._obb_dict[name] = np.ascontiguousarray(corners_8x3, dtype=np.float32)
        self.update()

    def clear_obb(self, name=None):
        """OBB 제거. name=None이면 전체."""
        if name is None:
            self._obb_dict.clear()
        else:
            self._obb_dict.pop(name, None)
        self.update()

    def set_view(self, azimuth, elevation):
        self._azimuth = azimuth
        self._elevation = elevation
        self.update()

    def set_top_view(self):
        self.set_view(0.0, 90.0)

    def set_front_view(self):
        self.set_view(0.0, 0.0)

    def set_right_view(self):
        self.set_view(90.0, 0.0)

    # ── Helpers ──

    def _calc_clip_planes(self):
        return self._clip_near, self._clip_far

    def _fb_size(self):
        dpr = self.devicePixelRatioF()
        return int(self.width() * dpr), int(self.height() * dpr)

    # ── OpenGL lifecycle ──

    def initializeGL(self):
        glClearColor(0.15, 0.15, 0.18, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)
        self._init_shaders()

    def resizeGL(self, w, h):
        fw, fh = self._fb_size()
        glViewport(0, 0, fw, fh)

    def paintGL(self):
        near, far = self._calc_clip_planes()
        fw, fh = self._fb_size()

        edl_ok = (self._edl_enabled and self._has_visible_points()
                  and self._edl_depth_shader is not None
                  and self._edl_shade_shader is not None)

        if edl_ok:
            self._paint_edl(near, far, fw, fh)
        else:
            self._paint_normal(near, far, fw, fh)

        # 메시 + Slice 평면 (depth test 유지, 반투명)
        self._setup_projection(near, far)
        self._setup_modelview()
        self._draw_mesh()
        if self._slice_planes:
            self._draw_slice_planes()

        # OBB + 폴리라인 + segment강조 + 마커 + 측정 + 축 (항상 위에)
        glDisable(GL_DEPTH_TEST)
        self._draw_obb()
        self._draw_highlight_seg()
        self._draw_marker()
        self._draw_measure()
        if self._guide_polylines:
            self._draw_guide_polylines()
        if self._guide_points:
            self._draw_guide_points()
        if self._polyline is not None:
            self._draw_polyline()
        if self._show_world_axes:
            self._draw_world_axes()
        if self._axes:
            self._draw_pca_axes()
        glEnable(GL_DEPTH_TEST)

        # 행렬 캐시 (마우스 이벤트에서 사용)
        self._last_modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._last_projection = glGetDoublev(GL_PROJECTION_MATRIX)
        self._last_viewport = glGetIntegerv(GL_VIEWPORT)

        self._draw_axis_labels()

    def _paint_normal(self, near, far, fw, fh):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        self._setup_projection(near, far)
        self._setup_modelview()
        if self._has_visible_points():
            if not self.blending:
                glDisable(GL_BLEND)
            if self.point_smooth and self._circle_shader:
                glUseProgram(self._circle_shader)
                glUniform1f(glGetUniformLocation(self._circle_shader, "uPointSize"), self.point_size)
            self._draw_points()
            glUseProgram(0)
            glEnable(GL_BLEND)

    def _paint_edl(self, near, far, fw, fh):
        # ── 1st pass: FBO에 점군 렌더 (alpha = fixedDepth) ──
        self._ensure_edl_fbo(fw, fh)
        glBindFramebuffer(GL_FRAMEBUFFER, self._edl_fbo)
        glViewport(0, 0, fw, fh)
        glClearColor(0.15, 0.15, 0.18, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        self._setup_projection(near, far)
        self._setup_modelview()

        if not self.blending:
            glDisable(GL_BLEND)
        glUseProgram(self._edl_depth_shader)
        glUniform1f(glGetUniformLocation(self._edl_depth_shader, "uPointSize"), self.point_size)
        glUniform1i(glGetUniformLocation(self._edl_depth_shader, "uCircle"),
                     1 if self.point_smooth else 0)
        self._draw_points()
        glUseProgram(0)
        glEnable(GL_BLEND)

        # ── 2nd pass: EDL shade → 기본 FBO ──
        glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        glViewport(0, 0, fw, fh)
        glClearColor(0.15, 0.15, 0.18, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._draw_edl_pass(fw, fh)

    # ── Projection / Modelview ──

    def _setup_projection(self, near, far):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width() / max(self.height(), 1)
        if self._perspective:
            gluPerspective(45.0, aspect, near, far)
        else:
            half_h = self._distance * 0.5
            half_w = half_h * aspect
            glOrtho(-half_w, half_w, -half_h, half_h, near, far)
        glMatrixMode(GL_MODELVIEW)

    def _setup_modelview(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -self._distance)
        # 스크린 좌표계 패닝 (카메라 회전 이후에 적용)
        glTranslatef(self._pan_screen[0], self._pan_screen[1], 0)
        glRotatef(self._elevation, 1, 0, 0)
        glRotatef(self._azimuth, 0, 1, 0)
        if self._convention == AxisConvention.ENU:
            glRotatef(-90, 1, 0, 0)
        # 월드 좌표 중심 오프셋
        glTranslatef(self._center[0], self._center[1], self._center[2])

    # ── EDL shaders / FBO ──

    def _init_shaders(self):
        try:
            # 원형 포인트 셰이더 (일반 렌더링용)
            vs0 = gl_shaders.compileShader(_CIRCLE_VERT_SRC, GL_VERTEX_SHADER)
            fs0 = gl_shaders.compileShader(_CIRCLE_FRAG_SRC, GL_FRAGMENT_SHADER)
            self._circle_shader = gl_shaders.compileProgram(vs0, fs0)

            # EDL 1st pass
            vs1 = gl_shaders.compileShader(_DEPTH_VERT_SRC, GL_VERTEX_SHADER)
            fs1 = gl_shaders.compileShader(_DEPTH_FRAG_SRC, GL_FRAGMENT_SHADER)
            self._edl_depth_shader = gl_shaders.compileProgram(vs1, fs1)

            # EDL 2nd pass
            vs2 = gl_shaders.compileShader(_EDL_VERT_SRC, GL_VERTEX_SHADER)
            fs2 = gl_shaders.compileShader(_EDL_FRAG_SRC, GL_FRAGMENT_SHADER)
            self._edl_shade_shader = gl_shaders.compileProgram(vs2, fs2)
            print("[Shaders] All compiled OK")
        except Exception as e:
            print(f"[Shaders] Compile failed: {e}")
            self._circle_shader = None
            self._edl_depth_shader = None
            self._edl_shade_shader = None

    def _ensure_edl_fbo(self, fw, fh):
        if self._edl_fbo is not None and self._edl_fbo_size == (fw, fh):
            return

        if self._edl_fbo is not None:
            glDeleteFramebuffers(1, [self._edl_fbo])
            glDeleteTextures([self._edl_color_tex])
            glDeleteRenderbuffers(1, [self._edl_depth_rb])

        # Color+depth texture (RGBA32F: RGB=색상, A=fixedDepth)
        self._edl_color_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._edl_color_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, fw, fh, 0,
                     GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Depth renderbuffer (Z-test 용)
        self._edl_depth_rb = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self._edl_depth_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, fw, fh)

        self._edl_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._edl_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, self._edl_color_tex, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, self._edl_depth_rb)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"[EDL] FBO incomplete: {status:#x}")
            glDeleteFramebuffers(1, [self._edl_fbo])
            self._edl_fbo = None

        glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        self._edl_fbo_size = (fw, fh)

    def _draw_edl_pass(self, fw, fh):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glUseProgram(self._edl_shade_shader)

        loc = lambda name: glGetUniformLocation(self._edl_shade_shader, name)
        glUniform1i(loc("uColorTex"), 0)
        glUniform1f(loc("uScreenWidth"), float(fw))
        glUniform1f(loc("uScreenHeight"), float(fh))
        glUniform1f(loc("uRadius"), self._edl_radius)
        glUniform1f(loc("uStrength"), self._edl_strength)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._edl_color_tex)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f( 1, -1)
        glTexCoord2f(1, 1); glVertex2f( 1,  1)
        glTexCoord2f(0, 1); glVertex2f(-1,  1)
        glEnd()

        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    # ── Point / axis rendering ──

    def _has_visible_points(self):
        return any(l.visible and l.count > 0 for l in self._layers)

    def _draw_points(self):
        glPointSize(self.point_size)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        for layer in self._layers:
            if not layer.visible or layer.count == 0:
                continue
            glVertexPointer(3, GL_FLOAT, 0, layer.vertices)
            glColorPointer(3, GL_FLOAT, 0, layer.colors)
            glDrawArrays(GL_POINTS, 0, layer.count)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def _get_visible_bbox_center(self):
        """visible 레이어의 bbox 중심."""
        all_pts = [l.vertices for l in self._layers if l.visible and l.count > 0]
        if not all_pts:
            return np.zeros(3)
        merged = np.vstack(all_pts)
        return (merged.min(axis=0) + merged.max(axis=0)) * 0.5

    def _draw_world_axes(self):
        c = self._get_visible_bbox_center()
        axis_len = self._distance * 0.15
        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glColor3f(0.6, 0.2, 0.2); glVertex3f(cx, cy, cz); glVertex3f(cx + axis_len, cy, cz)
        glColor3f(0.2, 0.6, 0.2); glVertex3f(cx, cy, cz); glVertex3f(cx, cy + axis_len, cz)
        glColor3f(0.2, 0.2, 0.6); glVertex3f(cx, cy, cz); glVertex3f(cx, cy, cz + axis_len)
        glEnd()

    def _draw_pca_axes(self):
        glLineWidth(3.0)
        for origin, direction, color, label in self._axes:
            length = self._distance * 0.4
            end = origin + direction * length
            glBegin(GL_LINES)
            glColor3f(*color)
            glVertex3f(*origin); glVertex3f(*end)
            glEnd()
            arrow_len = length * 0.08
            perp = np.cross(direction, [0, 0, 1])
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(direction, [0, 1, 0])
            perp = perp / np.linalg.norm(perp) * arrow_len
            glBegin(GL_LINES)
            glColor3f(*color)
            glVertex3f(*end); glVertex3f(*(end - direction * arrow_len * 2 + perp))
            glVertex3f(*end); glVertex3f(*(end - direction * arrow_len * 2 - perp))
            glEnd()

    def _draw_slice_planes(self):
        """Slice 평면 시각화: 비활성=얇은 선, 활성=반투명 사각형."""
        if not self._slice_planes:
            return

        active_y = self._active_slice[0] if self._active_slice else None
        active_thick = self._active_slice[1] if self._active_slice else 0

        for y, xmin, xmax, zmin, zmax in self._slice_planes:
            is_active = (active_y is not None and abs(y - active_y) < 0.001)

            if is_active:
                # 활성 slice: 반투명 사각형 + 두께 범위 표시
                half_t = active_thick * 0.5
                glColor4f(1.0, 0.8, 0.0, 0.15)
                glBegin(GL_QUADS)
                glVertex3f(xmin, y - half_t, zmin)
                glVertex3f(xmax, y - half_t, zmin)
                glVertex3f(xmax, y + half_t, zmin)
                glVertex3f(xmin, y + half_t, zmin)

                glVertex3f(xmin, y - half_t, zmax)
                glVertex3f(xmax, y - half_t, zmax)
                glVertex3f(xmax, y + half_t, zmax)
                glVertex3f(xmin, y + half_t, zmax)

                glVertex3f(xmin, y - half_t, zmin)
                glVertex3f(xmin, y + half_t, zmin)
                glVertex3f(xmin, y + half_t, zmax)
                glVertex3f(xmin, y - half_t, zmax)

                glVertex3f(xmax, y - half_t, zmin)
                glVertex3f(xmax, y + half_t, zmin)
                glVertex3f(xmax, y + half_t, zmax)
                glVertex3f(xmax, y - half_t, zmax)
                glEnd()

                # 중심선
                glLineWidth(2.0)
                glColor4f(1.0, 1.0, 0.0, 0.8)
                glBegin(GL_LINE_LOOP)
                glVertex3f(xmin, y, zmin)
                glVertex3f(xmax, y, zmin)
                glVertex3f(xmax, y, zmax)
                glVertex3f(xmin, y, zmax)
                glEnd()
            else:
                # 비활성 slice: 얇은 선
                glLineWidth(1.0)
                glColor4f(0.5, 0.5, 0.5, 0.3)
                glBegin(GL_LINE_LOOP)
                glVertex3f(xmin, y, zmin)
                glVertex3f(xmax, y, zmin)
                glVertex3f(xmax, y, zmax)
                glVertex3f(xmin, y, zmax)
                glEnd()

    def _draw_polyline(self):
        if self._polyline is None or len(self._polyline) < 3:
            return
        # 라인
        glLineWidth(2.5)
        glColor3f(1.0, 1.0, 0.0)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self._polyline)
        glDrawArrays(GL_LINE_LOOP, 0, len(self._polyline))
        # vertex 포인트
        if self._polyline_vtx_visible:
            glPointSize(self._polyline_vtx_size)
            glColor3f(1.0, 0.3, 0.0)  # 주황색
            glDrawArrays(GL_POINTS, 0, len(self._polyline))
        glDisableClientState(GL_VERTEX_ARRAY)

    def _draw_guide_polylines(self):
        glLineWidth(1.2)
        glEnableClientState(GL_VERTEX_ARRAY)
        for points, color in self._guide_polylines:
            if len(points) < 2:
                continue
            glColor3f(float(color[0]), float(color[1]), float(color[2]))
            glVertexPointer(3, GL_FLOAT, 0, points)
            glDrawArrays(GL_LINE_LOOP, 0, len(points))
        glDisableClientState(GL_VERTEX_ARRAY)

    def _draw_guide_points(self):
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        for point, color, size, _label in self._guide_points:
            glPointSize(float(size))
            glColor3f(float(color[0]), float(color[1]), float(color[2]))
            glBegin(GL_POINTS)
            glVertex3f(float(point[0]), float(point[1]), float(point[2]))
            glEnd()
        glDisable(GL_POINT_SMOOTH)

    def _draw_mesh(self):
        if self._mesh_vertices is None or self._mesh_count == 0 or not self._mesh_visible:
            return
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0.3, 0.5, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.4, 0.7, 0.9, 0.6])
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.2, 0.3, 0.4, 0.6])
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self._mesh_vertices)
        glNormalPointer(GL_FLOAT, 0, self._mesh_normals)
        glDrawArrays(GL_TRIANGLES, 0, self._mesh_count)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisable(GL_LIGHTING)
        glDisable(GL_LIGHT0)

    def _draw_measure(self):
        pts = self._measure_points
        if len(pts) < 1:
            return
        # 점 표시
        glPointSize(8.0)
        glColor3f(0.0, 1.0, 0.0)  # 초록색 점
        glBegin(GL_POINTS)
        for p in pts:
            glVertex3f(*p)
        glEnd()
        # 선분 표시
        if len(pts) >= 2:
            glLineWidth(2.0)
            glColor3f(0.0, 1.0, 0.0)
            glBegin(GL_LINE_STRIP)
            for p in pts:
                glVertex3f(*p)
            glEnd()

    def _draw_measure_labels(self, painter):
        """측정 거리 텍스트 표시."""
        pts = self._measure_points
        if len(pts) < 2:
            return
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor(0, 255, 0))
        total_dist = 0.0
        for i in range(1, len(pts)):
            p0 = pts[i - 1]
            p1 = pts[i]
            d = np.linalg.norm(p1 - p0)
            total_dist += d
            mid = (p0 + p1) * 0.5
            sx, sy = self._project_to_screen(*mid)
            painter.drawText(sx + 5, sy - 5, f"{d:.3f}m")
        # 총 거리
        if len(pts) > 2:
            sx, sy = self._project_to_screen(*pts[-1])
            painter.setPen(QColor(255, 255, 0))
            painter.drawText(sx + 5, sy + 15, f"총: {total_dist:.3f}m")

    def _draw_highlight_seg(self):
        if self._highlight_seg is None:
            return
        p1, p2 = self._highlight_seg[0], self._highlight_seg[1]
        # 두꺼운 시안 선분
        glLineWidth(4.0)
        glColor3f(0.0, 1.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(*p1)
        glVertex3f(*p2)
        glEnd()
        # 양 끝점
        glPointSize(10.0)
        glColor3f(0.0, 1.0, 1.0)
        glBegin(GL_POINTS)
        glVertex3f(*p1)
        glVertex3f(*p2)
        glEnd()

    def _draw_marker(self):
        if self._marker_pos is None:
            return
        x, y, z = self._marker_pos
        s = self._marker_size
        glLineWidth(2.0)
        glColor3f(1.0, 0.0, 0.0)  # 빨간색 박스
        # 사각형 (XZ 평면 — 단면 뷰 기준)
        glBegin(GL_LINE_LOOP)
        glVertex3f(x - s, y, z - s)
        glVertex3f(x + s, y, z - s)
        glVertex3f(x + s, y, z + s)
        glVertex3f(x - s, y, z + s)
        glEnd()
        # 십자
        glBegin(GL_LINES)
        glVertex3f(x - s, y, z)
        glVertex3f(x + s, y, z)
        glVertex3f(x, y, z - s)
        glVertex3f(x, y, z + s)
        glEnd()

    def _draw_obb(self):
        if not self._obb_visible or not self._obb_dict:
            return
        # 레이어 visible 상태와 연동
        visible_layers = {l.name for l in self._layers if l.visible}
        for name, corners in self._obb_dict.items():
            if name not in visible_layers:
                continue
            c = corners
            glLineWidth(1.5)
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_LINE_LOOP)
            for i in range(4):
                glVertex3f(*c[i])
            glEnd()
            glBegin(GL_LINE_LOOP)
            for i in range(4, 8):
                glVertex3f(*c[i])
            glEnd()
            glBegin(GL_LINES)
            for i in range(4):
                glVertex3f(*c[i])
                glVertex3f(*c[i + 4])
            glEnd()

    def _project_to_screen(self, x, y, z):
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        sx, sy, sz = gluProject(x, y, z, modelview, projection, viewport)
        dpr = self.devicePixelRatioF()
        return int(sx / dpr), int(self.height() - sy / dpr)

    def _draw_axis_labels(self):
        painter = QPainter(self)
        font = QFont("Arial", 11, QFont.Bold)
        painter.setFont(font)
        c = self._get_visible_bbox_center()
        axis_len = self._distance * 0.15
        labels = [
            (c[0] + axis_len * 1.08, c[1], c[2], "X", QColor(180, 60, 60)),
            (c[0], c[1] + axis_len * 1.08, c[2], "Y", QColor(60, 180, 60)),
            (c[0], c[1], c[2] + axis_len * 1.08, "Z", QColor(60, 60, 180)),
        ]
        for wx, wy, wz, text, color in labels:
            sx, sy = self._project_to_screen(wx, wy, wz)
            painter.setPen(color)
            painter.drawText(sx - 6, sy + 5, text)
        if self._axes:
            for origin, direction, color, label in self._axes:
                length = self._distance * 0.4
                end = origin + direction * length * 1.05
                sx, sy = self._project_to_screen(*end)
                r, g, b = [int(c * 255) for c in color]
                painter.setPen(QColor(r, g, b))
                painter.drawText(sx + 5, sy - 5, label)
        if self._guide_points:
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            for point, color, _size, label in self._guide_points:
                if not label:
                    continue
                sx, sy = self._project_to_screen(float(point[0]), float(point[1]), float(point[2]))
                r, g, b = [int(max(0.0, min(1.0, float(c))) * 255) for c in color[:3]]
                painter.setPen(QColor(r, g, b))
                painter.drawText(sx + 8, sy - 8, label)
        # 측정 거리 라벨
        if self._measure_points:
            self._draw_measure_labels(painter)
        painter.end()

    # ── Mouse events ──

    def mousePressEvent(self, event):
        if self._measure_active and event.button() == Qt.LeftButton:
            pt = self._screen_to_world(event.x(), event.y())
            if pt is not None:
                self._measure_points.append(pt)
                self.update()
            return
        self._last_pos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self._last_pos.x()
        dy = event.y() - self._last_pos.y()
        if event.buttons() & Qt.LeftButton:
            self._azimuth += dx * 0.5
            self._elevation += dy * 0.5
            self._elevation = max(-90, min(90, self._elevation))
        elif event.buttons() & Qt.RightButton:
            scale = self._distance * 0.002
            self._pan_screen[0] += dx * scale
            self._pan_screen[1] -= dy * scale
        self._last_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self._distance *= factor
        self._distance = max(0.1, self._distance)
        self.update()


class ViewerPanel(QWidget):
    """Viewer3DWidget + 뷰 관련 툴바를 포함하는 패널."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gl = Viewer3DWidget(self)

        self._toolbar = QToolBar()
        self._toolbar.setMovable(False)
        self._toolbar.setIconSize(self._toolbar.iconSize())  # DPI 자동

        self._build_toolbar()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self.gl, 1)

    def _build_toolbar(self):
        tb = self._toolbar

        # 뷰 전환
        for name, shortcut, fn in [("Top", "7", self.gl.set_top_view),
                                    ("Front", "1", self.gl.set_front_view),
                                    ("Right", "3", self.gl.set_right_view)]:
            act = QAction(name, self)
            act.setShortcut(shortcut)
            act.triggered.connect(fn)
            tb.addAction(act)

        tb.addSeparator()

        # 투영 모드
        proj_group = QActionGroup(self)
        proj_group.setExclusive(True)
        self._act_persp = QAction("원근", self)
        self._act_persp.setCheckable(True)
        self._act_persp.setChecked(True)
        proj_group.addAction(self._act_persp)
        tb.addAction(self._act_persp)
        self._act_ortho = QAction("직교", self)
        self._act_ortho.setCheckable(True)
        proj_group.addAction(self._act_ortho)
        tb.addAction(self._act_ortho)
        proj_group.triggered.connect(
            lambda a: setattr(self.gl, 'perspective', a == self._act_persp) or self.gl.update()
        )

        tb.addSeparator()

        # EDL
        self._act_edl = QAction("EDL", self)
        self._act_edl.setCheckable(True)
        self._act_edl.setChecked(True)
        self._act_edl.toggled.connect(self._on_edl_toggled)
        tb.addAction(self._act_edl)
        self.gl.edl_enabled = True

        self._edl_label = QLabel(" 1.0 ")
        self._edl_slider = QSlider(Qt.Horizontal)
        self._edl_slider.setRange(1, 50)
        self._edl_slider.setValue(10)
        self._edl_slider.setFixedWidth(80)
        self._edl_slider.valueChanged.connect(self._on_edl_strength)
        tb.addWidget(self._edl_slider)
        tb.addWidget(self._edl_label)

        tb.addSeparator()

        # 점 모양 / 블렌딩
        act_circle = QAction("●", self)
        act_circle.setCheckable(True)
        act_circle.setToolTip("원형 포인트")
        act_circle.toggled.connect(lambda v: setattr(self.gl, 'point_smooth', v) or self.gl.update())
        tb.addAction(act_circle)

        act_blend = QAction("Blend", self)
        act_blend.setCheckable(True)
        act_blend.setToolTip("블렌딩 (반투명)")
        act_blend.toggled.connect(lambda v: setattr(self.gl, 'blending', v) or self.gl.update())
        tb.addAction(act_blend)

        tb.addSeparator()

        # 색상 모드
        color_group = QActionGroup(self)
        color_group.setExclusive(True)
        self._act_height = QAction("높이", self)
        self._act_height.setCheckable(True)
        self._act_height.setChecked(True)
        color_group.addAction(self._act_height)
        tb.addAction(self._act_height)
        self._act_rgb = QAction("RGB", self)
        self._act_rgb.setCheckable(True)
        color_group.addAction(self._act_rgb)
        tb.addAction(self._act_rgb)
        # 색상 변경은 mainwindow에서 연결 (데이터 접근 필요)

        tb.addSeparator()

        # Point size
        tb.addWidget(QLabel(" Pt: "))
        self._pt_label = QLabel(" 2 ")
        self._pt_slider = QSlider(Qt.Horizontal)
        self._pt_slider.setRange(1, 30)
        self._pt_slider.setValue(2)
        self._pt_slider.setFixedWidth(100)
        self._pt_slider.valueChanged.connect(self._on_pt_size)
        tb.addWidget(self._pt_slider)
        tb.addWidget(self._pt_label)

    # ── 내부 핸들러 ──

    def _on_edl_toggled(self, checked):
        self.gl.edl_enabled = checked

    def _on_edl_strength(self, value):
        s = value / 10.0
        self.gl._edl_strength = s
        self._edl_label.setText(f" {s:.1f} ")
        if self.gl.edl_enabled:
            self.gl.update()

    def _on_pt_size(self, value):
        self.gl.point_size = float(value)
        self._pt_label.setText(f" {value} ")
        self.gl.update()

    # ── GL 위임 (mainwindow 호환) ──

    def __getattr__(self, name):
        """ViewerPanel에 없는 속성은 gl 위젯으로 위임."""
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self.gl, name)
