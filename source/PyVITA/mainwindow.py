import os
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QToolBar, QAction, QFileDialog, QApplication,
    QStatusBar, QMessageBox, QSlider, QLabel, QComboBox, QActionGroup,
    QDockWidget, QWidget, QVBoxLayout, QListWidget, QPushButton,
    QHBoxLayout, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QProgressDialog, QCheckBox, QLineEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor, QColor, QBrush, QDoubleValidator
from viewer3d import ViewerPanel
from pointcloud import PointCloud, AxisConvention, SUPPORTED_EXTENSIONS
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySamsLab - Tunnel Shotcrete Monitor")
        self.resize(1200, 800)
        self.pc = PointCloud()
        self.pc_aligned = None
        self.viewer_panel = ViewerPanel()
        self.viewer = self.viewer_panel.gl  # GL 위젯 직접 참조
        self.setCentralWidget(self.viewer_panel)
        # 색상 모드 시그널 연결 (데이터 접근 필요하므로 mainwindow에서)
        self.viewer_panel._act_height.triggered.connect(lambda: self._apply_color('height'))
        self.viewer_panel._act_rgb.triggered.connect(lambda: self._apply_color('rgb'))
        self._slices = []
        self._section_active = False
        self._current_section = None
        self._create_toolbar()
        self._create_slice_dock()
        self._create_segment_dock()
        self._create_statusbar()
        self.setAcceptDrops(True)
    def _create_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        # 파일 열기
        act_open = QAction("파일 열기", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_file)
        toolbar.addAction(act_open)
        toolbar.addSeparator()
        # 좌표계 선택
        toolbar.addWidget(QLabel(" 좌표계: "))
        self._conv_combo = QComboBox()
        self._conv_combo.addItem("ENU (Z-Up)", AxisConvention.ENU)
        self._conv_combo.addItem("EDN (Y-Up)", AxisConvention.EDN)
        self._conv_combo.currentIndexChanged.connect(self._on_convention_changed)
        toolbar.addWidget(self._conv_combo)
        toolbar.addSeparator()
        # 방향 잡기
        self._axis_method_combo = QComboBox()
        self._axis_method_combo.addItem("PCA", "PCA")
        self._axis_method_combo.addItem("OBB", "OBB")
        self._axis_method_combo.setCurrentIndex(1)
        toolbar.addWidget(self._axis_method_combo)
        self.act_find_axes = QAction("방향 잡기", self)
        self.act_find_axes.setEnabled(False)
        self.act_find_axes.triggered.connect(self._find_axes)
        toolbar.addAction(self.act_find_axes)
        self.act_rot_cw = QAction("↻90", self)
        self.act_rot_cw.setToolTip("W축 기준 시계방향 90° 회전")
        self.act_rot_cw.setEnabled(False)
        self.act_rot_cw.triggered.connect(lambda: self._rotate_axes(90))
        toolbar.addAction(self.act_rot_cw)
        self.act_rot_ccw = QAction("↺90", self)
        self.act_rot_ccw.setToolTip("W축 기준 반시계방향 90° 회전")
        self.act_rot_ccw.setEnabled(False)
        self.act_rot_ccw.triggered.connect(lambda: self._rotate_axes(-90))
        toolbar.addAction(self.act_rot_ccw)
        self._act_obb_box = QAction("OBB", self)
        self._act_obb_box.setCheckable(True)
        self._act_obb_box.setChecked(True)
        self._act_obb_box.setEnabled(False)
        self._act_obb_box.setToolTip("OBB 박스 표시")
        self._act_obb_box.toggled.connect(self._on_obb_box_toggled)
        toolbar.addAction(self._act_obb_box)
        self.act_align = QAction("정렬", self)
        self.act_align.setEnabled(False)
        self.act_align.triggered.connect(self._align)
        toolbar.addAction(self.act_align)
        toolbar.addSeparator()
        # 레이어 show/hide
        self._chk_original = QAction("원본", self)
        self._chk_original.setCheckable(True)
        self._chk_original.setChecked(True)
        self._chk_original.setEnabled(False)
        self._chk_original.toggled.connect(lambda v: self.viewer.set_layer_visible("원본", v))
        toolbar.addAction(self._chk_original)
        self._chk_aligned = QAction("정렬", self)
        self._chk_aligned.setCheckable(True)
        self._chk_aligned.setChecked(False)
        self._chk_aligned.setEnabled(False)
        self._chk_aligned.toggled.connect(lambda v: self.viewer.set_layer_visible("정렬", v))
        toolbar.addAction(self._chk_aligned)
    def _create_slice_dock(self):
        dock = QDockWidget("Slice", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("간격:"))
        self._spin_interval = QDoubleSpinBox()
        self._spin_interval.setRange(0.1, 5.0)
        self._spin_interval.setValue(1.0)
        self._spin_interval.setSingleStep(0.1)
        self._spin_interval.setSuffix(" m")
        param_layout.addWidget(self._spin_interval)
        param_layout.addWidget(QLabel("두께:"))
        self._spin_thickness = QDoubleSpinBox()
        self._spin_thickness.setRange(0.05, 5.0)
        self._spin_thickness.setValue(1.0)
        self._spin_thickness.setSingleStep(0.01)
        self._spin_thickness.setSuffix(" m")
        param_layout.addWidget(self._spin_thickness)
        layout.addLayout(param_layout)
        # Slice 생성 / 삭제
        slice_btn_layout = QHBoxLayout()
        self._btn_create_slices = QPushButton("Slice 생성")
        self._btn_create_slices.setEnabled(False)
        self._btn_create_slices.clicked.connect(self._create_slices)
        slice_btn_layout.addWidget(self._btn_create_slices)
        self._btn_clear_slices = QPushButton("Slice 삭제")
        self._btn_clear_slices.setEnabled(False)
        self._btn_clear_slices.clicked.connect(self._clear_slices)
        slice_btn_layout.addWidget(self._btn_clear_slices)
        layout.addLayout(slice_btn_layout)
        self._slice_list = QListWidget()
        self._slice_list.currentRowChanged.connect(self._on_slice_selected)
        self._slice_list.itemDoubleClicked.connect(lambda: self._show_section())
        layout.addWidget(self._slice_list)
        btn_layout = QHBoxLayout()
        self._btn_show_section = QPushButton("단면 보기")
        self._btn_show_section.setEnabled(False)
        self._btn_show_section.clicked.connect(self._show_section)
        btn_layout.addWidget(self._btn_show_section)
        self._btn_close_section = QPushButton("단면 닫기")
        self._btn_close_section.setEnabled(False)
        self._btn_close_section.clicked.connect(self._close_section)
        btn_layout.addWidget(self._btn_close_section)
        layout.addLayout(btn_layout)
        # 측정 도구
        measure_layout = QHBoxLayout()
        self._btn_measure = QPushButton("📏 거리 측정")
        self._btn_measure.setCheckable(True)
        self._btn_measure.setEnabled(False)
        self._btn_measure.toggled.connect(self._on_measure_toggled)
        measure_layout.addWidget(self._btn_measure)
        self._btn_measure_clear = QPushButton("측정 초기화")
        self._btn_measure_clear.setEnabled(False)
        self._btn_measure_clear.clicked.connect(self._on_measure_clear)
        measure_layout.addWidget(self._btn_measure_clear)
        layout.addLayout(measure_layout)
        # ── 외곽 추출 영역 ──
        # 방식 선택
        contour_layout = QHBoxLayout()
        contour_layout.addWidget(QLabel("외곽:"))
        self._contour_combo = QComboBox()
        self._contour_combo.addItem("Radial", "radial")
        self._contour_combo.addItem("Convex Hull", "convex")
        self._contour_combo.addItem("Concave Hull", "concave")
        self._contour_combo.addItem("Alpha Shape", "alpha")
        self._contour_combo.addItem("Top Circle", "circle")
        self._contour_combo.addItem("@ Grid + Marching Squares", "grid_ms")
        self._contour_combo.currentIndexChanged.connect(self._on_contour_method_changed)
        contour_layout.addWidget(self._contour_combo)
        layout.addLayout(contour_layout)
        # 파라미터 (방식에 따라 표시/숨김)
        self._contour_param_layout = QHBoxLayout()
        self._contour_param_label = QLabel("파라미터:")
        self._contour_param_spin = QDoubleSpinBox()
        self._contour_param_spin.setRange(0.01, 100.0)
        self._contour_param_spin.setValue(2.0)
        self._contour_param_spin.setSingleStep(0.1)
        self._contour_param_label2 = QLabel("파라미터2:")
        self._contour_param_spin2 = QDoubleSpinBox()
        self._contour_param_spin2.setRange(1, 1000)
        self._contour_param_spin2.setValue(360)
        self._contour_param_spin2.setSingleStep(1)
        self._contour_param_layout.addWidget(self._contour_param_label)
        self._contour_param_layout.addWidget(self._contour_param_spin)
        self._contour_param_layout.addWidget(self._contour_param_label2)
        self._contour_param_layout.addWidget(self._contour_param_spin2)
        layout.addLayout(self._contour_param_layout)
        self._contour_param_label.hide()
        self._contour_param_spin.hide()
        self._contour_param_label2.hide()
        self._contour_param_spin2.hide()
        radial_min_layout = QHBoxLayout()
        self._chk_radial_min_radius = QCheckBox("최소 반경")
        self._chk_radial_min_radius.setChecked(True)
        self._edit_radial_min_radius = QLineEdit("1.0")
        self._edit_radial_min_radius.setValidator(QDoubleValidator(0.0, 10000.0, 3, self))
        self._edit_radial_min_radius.setFixedWidth(64)
        self._edit_radial_min_radius.setEnabled(True)
        self._chk_radial_min_radius.toggled.connect(self._edit_radial_min_radius.setEnabled)
        radial_min_layout.addWidget(self._chk_radial_min_radius)
        radial_min_layout.addWidget(self._edit_radial_min_radius)
        self._label_radial_min_radius_unit = QLabel("m")
        radial_min_layout.addWidget(self._label_radial_min_radius_unit)
        radial_min_layout.addStretch(1)
        layout.addLayout(radial_min_layout)
        self._chk_radial_min_radius.hide()
        self._edit_radial_min_radius.hide()
        self._label_radial_min_radius_unit.hide()
        radial_percentile_layout = QHBoxLayout()
        self._chk_radial_percentile = QCheckBox("Percentile")
        self._chk_radial_percentile.setChecked(False)
        self._spin_radial_percentile = QDoubleSpinBox()
        self._spin_radial_percentile.setRange(50.0, 100.0)
        self._spin_radial_percentile.setValue(98.0)
        self._spin_radial_percentile.setSingleStep(0.5)
        self._spin_radial_percentile.setDecimals(1)
        self._spin_radial_percentile.setSuffix(" %")
        self._spin_radial_percentile.setEnabled(False)
        radial_percentile_layout.addWidget(self._chk_radial_percentile)
        radial_percentile_layout.addWidget(self._spin_radial_percentile)
        radial_percentile_layout.addStretch(1)
        layout.addLayout(radial_percentile_layout)
        self._chk_radial_percentile.hide()
        self._spin_radial_percentile.hide()
        radial_midline_layout = QHBoxLayout()
        self._chk_radial_midline = QCheckBox("Midline")
        self._chk_radial_midline.setChecked(False)
        self._spin_radial_midline_outer = QDoubleSpinBox()
        self._spin_radial_midline_outer.setRange(55.0, 99.5)
        self._spin_radial_midline_outer.setValue(90.0)
        self._spin_radial_midline_outer.setSingleStep(0.5)
        self._spin_radial_midline_outer.setDecimals(1)
        self._spin_radial_midline_outer.setSuffix(" %")
        self._spin_radial_midline_outer.setEnabled(False)
        radial_midline_layout.addWidget(self._chk_radial_midline)
        radial_midline_layout.addWidget(self._spin_radial_midline_outer)
        radial_midline_layout.addStretch(1)
        layout.addLayout(radial_midline_layout)
        self._chk_radial_midline.hide()
        self._spin_radial_midline_outer.hide()
        self._chk_radial_percentile.toggled.connect(self._on_radial_percentile_toggled)
        self._chk_radial_midline.toggled.connect(self._on_radial_midline_toggled)
        self._on_contour_method_changed(self._contour_combo.currentIndex())
        # 외곽 추출 버튼
        self._btn_extract_contour = QPushButton("외곽 추출")
        self._btn_extract_contour.setEnabled(False)
        self._btn_extract_contour.clicked.connect(self._extract_contour)
        layout.addWidget(self._btn_extract_contour)
        # vertex 표시 옵션
        vtx_layout = QHBoxLayout()
        self._chk_contour_vtx = QAction("Vertex", self)
        self._chk_contour_vtx.setCheckable(True)
        self._chk_contour_vtx.setChecked(True)
        self._chk_contour_vtx.toggled.connect(self._on_contour_vtx_toggled)
        vtx_btn = QPushButton("Vertex")
        vtx_btn.setCheckable(True)
        vtx_btn.setChecked(True)
        vtx_btn.toggled.connect(self._on_contour_vtx_toggled)
        self._btn_contour_vtx = vtx_btn
        vtx_layout.addWidget(vtx_btn)
        vtx_layout.addWidget(QLabel("크기:"))
        self._spin_vtx_size = QDoubleSpinBox()
        self._spin_vtx_size.setRange(1, 20)
        self._spin_vtx_size.setValue(5)
        self._spin_vtx_size.setSingleStep(1)
        self._spin_vtx_size.setDecimals(0)
        self._spin_vtx_size.valueChanged.connect(self._on_contour_vtx_size_changed)
        vtx_layout.addWidget(self._spin_vtx_size)
        layout.addLayout(vtx_layout)
        # 전체 외곽 + 면 생성
        self._btn_batch_surface = QPushButton("전체 외곽 면 생성")
        self._btn_batch_surface.setEnabled(False)
        self._btn_batch_surface.setToolTip("모든 slice의 외곽을 추출하고 extrusion하여 면 생성")
        self._btn_batch_surface.clicked.connect(self._batch_contour_surface)
        layout.addWidget(self._btn_batch_surface)
        dock.setWidget(w)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
    def _busy(self, busy=True):
        """시간이 걸리는 작업 시 커서 변경."""
        if busy:
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            QApplication.processEvents()
        else:
            QApplication.restoreOverrideCursor()
    def _create_segment_dock(self):
        """오른쪽 Segment 분석 패널."""
        dock = QDockWidget("Segments", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        # 파라미터
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("범위:"))
        self._spin_search_range = QDoubleSpinBox()
        self._spin_search_range.setRange(0.01, 5.0)
        self._spin_search_range.setValue(0.5)
        self._spin_search_range.setSingleStep(0.05)
        self._spin_search_range.setSuffix(" m")
        param_layout.addWidget(self._spin_search_range)
        param_layout.addWidget(QLabel("임계:"))
        self._spin_dist_threshold = QDoubleSpinBox()
        self._spin_dist_threshold.setRange(0.001, 10.0)
        self._spin_dist_threshold.setValue(0.1)
        self._spin_dist_threshold.setSingleStep(0.01)
        self._spin_dist_threshold.setSuffix(" m")
        param_layout.addWidget(self._spin_dist_threshold)
        layout.addLayout(param_layout)
        # Segment 테이블
        self._seg_table = QTableWidget()
        self._seg_table.setColumnCount(3)
        self._seg_table.setHorizontalHeaderLabels(["#", "길이", "최대거리"])
        header = self._seg_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.resizeSection(0, 40)
        self._seg_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._seg_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._seg_table.setSortingEnabled(True)
        self._seg_table.cellDoubleClicked.connect(self._on_segment_dblclicked)
        layout.addWidget(self._seg_table)
        # segment 분석 결과 캐시
        self._seg_analysis = []  # [{p1, p2, max_pt_3d, max_dist, seg_p1_3d, seg_p2_3d}, ...]
        dock.setWidget(w)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._segment_dock = dock
    def _create_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("LAS/PLY 파일을 열거나 드래그하세요.")
    # ── 설정 ──
    def _on_convention_changed(self, index):
        conv = self._conv_combo.itemData(index)
        self.pc.convention = conv
        self.viewer.convention = conv
    def _on_obb_box_toggled(self, checked):
        self.viewer._obb_visible = checked
        self.viewer.update()
    def _on_measure_toggled(self, checked):
        self.viewer.set_measure_active(checked)
        self._btn_measure_clear.setEnabled(checked)
    def _on_measure_clear(self):
        self.viewer.clear_measure()
    def _on_contour_vtx_toggled(self, checked):
        self.viewer._polyline_vtx_visible = checked
        self.viewer.update()
    def _on_contour_vtx_size_changed(self, value):
        self.viewer._polyline_vtx_size = float(value)
        self.viewer.update()
    def _on_radial_percentile_toggled(self, checked):
        self._spin_radial_percentile.setEnabled(checked)
        if checked and self._chk_radial_midline.isChecked():
            self._chk_radial_midline.setChecked(False)
    def _on_radial_midline_toggled(self, checked):
        self._spin_radial_midline_outer.setEnabled(checked)
        if checked and self._chk_radial_percentile.isChecked():
            self._chk_radial_percentile.setChecked(False)
    def _apply_color(self, mode):
        if self.pc.points is None:
            return
        if mode == 'rgb' and not self.pc.has_rgb:
            QMessageBox.information(self, "알림", "RGB 색상 정보가 없습니다.")
            self.viewer_panel._act_height.setChecked(True)
            return
        self.pc.apply_color_mode(mode)
        self.viewer.add_layer("원본", self.pc.points, self.pc.colors)
        self._chk_original.setChecked(True)
        if self.pc_aligned is not None:
            self.pc_aligned.colors = self.pc.colors.copy()
            self.viewer.add_layer("정렬", self.pc_aligned.points, self.pc_aligned.colors)
    # ── 파일 ──
    def _open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Point Cloud 파일 열기", "",
            "Point Cloud (*.las *.laz *.ply);;LAS Files (*.las *.laz);;PLY Files (*.ply);;All Files (*)"
        )
        if not filepath:
            return
        self._load_file(filepath)
    def _load_file(self, filepath):
        self._busy(True)
        try:
            count = self.pc.load(filepath)
        except Exception as e:
            self._busy(False)
            QMessageBox.critical(self, "오류", f"파일 로드 실패:\n{e}")
            return
        self.pc_aligned = None
        self._slices = []
        self._section_active = False
        self.viewer.clear_layers()
        self.viewer.add_layer("원본", self.pc.points, self.pc.colors)
        self.act_find_axes.setEnabled(True)
        self.act_align.setEnabled(False)
        self.act_rot_cw.setEnabled(False)
        self.act_rot_ccw.setEnabled(False)
        self._chk_original.setEnabled(True)
        self._chk_original.setChecked(True)
        self._chk_aligned.setEnabled(False)
        self._chk_aligned.setChecked(False)
        self._btn_create_slices.setEnabled(False)
        self._btn_show_section.setEnabled(False)
        self._btn_close_section.setEnabled(False)
        self._slice_list.clear()
        self._busy(False)
        filename = os.path.basename(filepath)
        self.statusbar.showMessage(
            f"{filename} | {count:,} points | "
            f"Offset: ({self.pc.offset[0]:.1f}, {self.pc.offset[1]:.1f}, {self.pc.offset[2]:.1f})"
        )
    # ── 드래그앤드롭 ──
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = os.path.splitext(url.toLocalFile())[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()
    def dropEvent(self, event):
        for url in event.mimeData().urls():
            filepath = url.toLocalFile()
            ext = os.path.splitext(filepath)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                self._load_file(filepath)
                return
    # ── 방향 잡기 / U↔V / 정렬 ──
    def _find_axes(self):
        self._busy(True)
        try:
            method = self._axis_method_combo.currentData()
            result = self.pc.find_axes(method=method)
            if result is None:
                return
            self._update_axes_display(method)
            # OBB 박스 표시
            if self.pc.obb_corners is not None:
                self.viewer.set_obb("원본", self.pc.obb_corners)
                self._act_obb_box.setEnabled(True)
                self._act_obb_box.setChecked(True)
            else:
                self.viewer.clear_obb()
                self._act_obb_box.setEnabled(False)
        finally:
            self._busy(False)
        self.act_align.setEnabled(True)
        self.act_rot_cw.setEnabled(True)
        self.act_rot_ccw.setEnabled(True)
    def _rotate_axes(self, deg):
        """W축(up) 기준으로 U/V를 deg도 회전."""
        if self.pc.primary_axis is None or self.pc.tertiary_axis is None:
            return
        rad = np.radians(deg)
        c, s = np.cos(rad), np.sin(rad)
        w = self.pc.tertiary_axis  # up축
        # Rodrigues 회전: v' = v*cos + (w x v)*sin + w*(w·v)*(1-cos)
        def rot(v):
            return v * c + np.cross(w, v) * s + w * np.dot(w, v) * (1 - c)
        self.pc.primary_axis = rot(self.pc.primary_axis)
        self.pc.secondary_axis = rot(self.pc.secondary_axis)
        # OBB 재계산
        conv = self.pc.convention
        horiz = [i for i in range(3) if i != conv.up_index]
        self.pc._compute_obb_from_axes(horiz, conv.up_index)
        method = self._axis_method_combo.currentData()
        self._update_axes_display(method)
        # OBB 갱신
        if self.pc.obb_corners is not None:
            self.viewer.set_obb("원본", self.pc.obb_corners)
    def _update_axes_display(self, method, swapped=False):
        primary = self.pc.primary_axis
        secondary = self.pc.secondary_axis
        tertiary = self.pc.tertiary_axis
        swap_tag = " [swapped]" if swapped else ""
        axes_list = [
            (primary,   (1.0, 1.0, 0.0), f"U ({method})"),
            (secondary, (0.0, 1.0, 1.0), f"V ({method})"),
            (tertiary,  (1.0, 0.0, 1.0), "W"),
        ]
        pmin, pmax = self.pc.get_bounds()
        bbox_center = (pmin + pmax) * 0.5
        self.viewer.set_axes(bbox_center, axes_list)
        ev = self.pc.eigenvalues
        if method == 'OBB':
            self.statusbar.showMessage(
                f"{method}{swap_tag} | U: [{primary[0]:.3f}, {primary[1]:.3f}, {primary[2]:.3f}] | "
                f"장축: {ev[0]:.2f}, 단축: {ev[1]:.2f}, 높이: {ev[2]:.2f}"
            )
        else:
            self.statusbar.showMessage(
                f"{method}{swap_tag} | U: [{primary[0]:.3f}, {primary[1]:.3f}, {primary[2]:.3f}] | "
                f"고유값 비: {ev[0]/ev.sum():.1%}, {ev[1]/ev.sum():.1%}, {ev[2]/ev.sum():.1%}"
            )
    def _align(self):
        self._busy(True)
        try:
            self._align_impl()
        finally:
            self._busy(False)
    def _align_impl(self):
        result = self.pc.make_aligned()
        if result is None:
            return
        aligned_points, aligned_colors = result
        conv = self.pc.convention
        self.pc_aligned = PointCloud()
        self.pc_aligned.points = aligned_points
        self.pc_aligned.colors = aligned_colors
        self.pc_aligned.centroid = np.mean(aligned_points, axis=0)
        self.pc_aligned.convention = conv
        self.viewer.add_layer("정렬", aligned_points, aligned_colors)
        self._chk_original.setChecked(False)
        self.viewer.set_layer_visible("원본", False)
        self._chk_aligned.setEnabled(True)
        self._chk_aligned.setChecked(True)
        # 정렬 레이어 OBB (AABB — 축 정렬 상태이므로)
        pmin = aligned_points.min(axis=0)
        pmax = aligned_points.max(axis=0)
        aligned_obb = np.array([
            [pmin[0], pmin[1], pmin[2]], [pmax[0], pmin[1], pmin[2]],
            [pmax[0], pmax[1], pmin[2]], [pmin[0], pmax[1], pmin[2]],
            [pmin[0], pmin[1], pmax[2]], [pmax[0], pmin[1], pmax[2]],
            [pmax[0], pmax[1], pmax[2]], [pmin[0], pmax[1], pmax[2]],
        ])
        self.viewer.set_obb("정렬", aligned_obb)
        fwd_label = "Y" if conv == AxisConvention.ENU else "Z"
        up_label = "Z" if conv == AxisConvention.ENU else "Y"
        axes_list = [
            (conv.forward_vector(), (1.0, 1.0, 0.0), f"U → {fwd_label}"),
            (conv.side_vector(),    (0.0, 1.0, 1.0), "V → X"),
            (conv.up_vector(),      (1.0, 0.0, 1.0), f"W → {up_label}"),
        ]
        pmin = aligned_points.min(axis=0)
        pmax = aligned_points.max(axis=0)
        bbox_center = (pmin + pmax) * 0.5
        self.viewer.set_axes(bbox_center, axes_list)
        self.viewer.set_top_view()
        self._btn_create_slices.setEnabled(True)
        self._slices = []
        self._slice_list.clear()
        self.viewer.clear_slices()
        self.statusbar.showMessage(
            f"정렬 완료 ({conv.value}) | U→{fwd_label}, V→X, W→{up_label}"
        )
    # ── Slice ──
    def _create_slices(self):
        if self.pc_aligned is None:
            return
        self._busy(True)
        try:
            self._create_slices_impl()
        finally:
            self._busy(False)
    def _create_slices_impl(self):
        if self.pc_aligned is None:
            return
        conv = self.pc_aligned.convention
        fwd_idx = conv.forward_index
        pts = self.pc_aligned.points
        y_min = pts[:, fwd_idx].min()
        y_max = pts[:, fwd_idx].max()
        interval = self._spin_interval.value()
        self._slices = []
        y = y_min + interval * 0.5
        while y < y_max:
            self._slices.append(y)
            y += interval
        self._slice_list.clear()
        for i, y_pos in enumerate(self._slices):
            self._slice_list.addItem(f"Slice {i}  (Y= {y_pos:.2f})")
        side_idx = conv.side_index
        up_idx = conv.up_index
        x_range = (pts[:, side_idx].min(), pts[:, side_idx].max())
        z_range = (pts[:, up_idx].min(), pts[:, up_idx].max())
        self.viewer.set_slice_planes(self._slices, x_range, z_range)
        self._btn_show_section.setEnabled(len(self._slices) > 0)
        self._btn_clear_slices.setEnabled(len(self._slices) > 0)
        self._btn_batch_surface.setEnabled(len(self._slices) > 0)
        self.statusbar.showMessage(f"{len(self._slices)}개 slice 생성 (간격: {interval}m)")
    def _clear_slices(self):
        """생성된 slice 전체 삭제."""
        self._slices = []
        self._slice_list.clear()
        self._btn_batch_surface.setEnabled(False)
        self.viewer.remove_layer("외곽면")
        self.viewer.clear_slices()
        self.viewer.clear_polyline()
        self._btn_show_section.setEnabled(False)
        self._btn_clear_slices.setEnabled(False)
        self._btn_extract_contour.setEnabled(False)
        if self._section_active:
            self._close_section()
        self.statusbar.showMessage("Slice 삭제 완료")
    def _on_contour_method_changed(self, index):
        """외곽 방식 변경 시 파라미터 표시/숨김."""
        method = self._contour_combo.currentData()
        self._contour_param_label2.hide()
        self._contour_param_spin2.hide()
        self._chk_radial_min_radius.hide()
        self._edit_radial_min_radius.hide()
        self._label_radial_min_radius_unit.hide()
        self._chk_radial_percentile.hide()
        self._spin_radial_percentile.hide()
        self._chk_radial_midline.hide()
        self._spin_radial_midline_outer.hide()
        if method == 'concave':
            self._contour_param_label.setText("Max Edge:")
            self._contour_param_spin.setRange(0.0, 100.0)
            self._contour_param_spin.setValue(1.0)
            self._contour_param_spin.setSingleStep(0.1)
            self._contour_param_spin.setDecimals(2)
            self._contour_param_label.show()
            self._contour_param_spin.show()
        elif method == 'alpha':
            self._contour_param_label.setText("Alpha:")
            self._contour_param_spin.setRange(0.0, 100.0)
            self._contour_param_spin.setValue(0.5)
            self._contour_param_spin.setSingleStep(0.5)
            self._contour_param_spin.setDecimals(1)
            self._contour_param_label.show()
            self._contour_param_spin.show()
        elif method == 'radial':
            self._contour_param_label.setText("Bins:")
            self._contour_param_spin.setRange(36, 720)
            self._contour_param_spin.setValue(360)
            self._contour_param_spin.setSingleStep(36)
            self._contour_param_spin.setDecimals(0)
            self._contour_param_label.show()
            self._contour_param_spin.show()
            self._chk_radial_min_radius.show()
            self._edit_radial_min_radius.show()
            self._label_radial_min_radius_unit.show()
            self._chk_radial_percentile.show()
            self._spin_radial_percentile.show()
            self._chk_radial_midline.show()
            self._spin_radial_midline_outer.show()
        elif method == 'grid_ms':
            self._contour_param_label.setText("Grid (m):")
            self._contour_param_spin.setRange(0.001, 0.50)
            self._contour_param_spin.setValue(0.05)
            self._contour_param_spin.setSingleStep(0.01)
            self._contour_param_spin.setDecimals(2)
            self._contour_param_label.show()
            self._contour_param_spin.show()
        elif method == 'circle':
            self._contour_param_label.setText("Points:")
            self._contour_param_spin.setRange(12, 360)
            self._contour_param_spin.setValue(120)
            self._contour_param_spin.setSingleStep(12)
            self._contour_param_spin.setDecimals(0)
            self._contour_param_label.show()
            self._contour_param_spin.show()
        else:
            self._contour_param_label.hide()
            self._contour_param_spin.hide()
    def _get_contour_kwargs(self, method):
        kwargs = {}
        if method == 'concave':
            kwargs['concavity'] = self._contour_param_spin.value()
        elif method == 'alpha':
            kwargs['alpha'] = self._contour_param_spin.value()
        elif method == 'radial':
            kwargs['num_bins'] = int(self._contour_param_spin.value())
            if self._chk_radial_min_radius.isChecked():
                try:
                    min_radius = float(self._edit_radial_min_radius.text())
                except ValueError:
                    min_radius = 1.0
                    self._edit_radial_min_radius.setText("1.0")
                kwargs['min_radius'] = max(0.0, min_radius)
            if self._chk_radial_percentile.isChecked():
                kwargs['percentile'] = self._spin_radial_percentile.value()
            if self._chk_radial_midline.isChecked():
                kwargs['midline'] = True
                kwargs['outer_percentile'] = self._spin_radial_midline_outer.value()
        elif method == 'grid_ms':
            kwargs['grid_size'] = self._contour_param_spin.value()
        elif method == 'circle':
            kwargs['num_pts'] = int(self._contour_param_spin.value())
        return kwargs
    def _on_slice_selected(self, row):
        if row < 0 or row >= len(self._slices):
            self.viewer.set_active_slice(0, 0)
            return
        y_pos = self._slices[row]
        thickness = self._spin_thickness.value()
        self.viewer.set_active_slice(y_pos, thickness)
        self._btn_show_section.setEnabled(True)
    def _show_section(self):
        row = self._slice_list.currentRow()
        if row < 0 or row >= len(self._slices) or self.pc_aligned is None:
            return
        y_pos = self._slices[row]
        thickness = self._spin_thickness.value()
        half_t = thickness * 0.5
        conv = self.pc_aligned.convention
        fwd_idx = conv.forward_index
        side_idx = conv.side_index
        up_idx = conv.up_index
        pts = self.pc_aligned.points
        colors = self.pc_aligned.colors
        mask = np.abs(pts[:, fwd_idx] - y_pos) <= half_t
        section_pts = pts[mask]
        section_colors = colors[mask]
        if len(section_pts) == 0:
            QMessageBox.information(self, "알림", "해당 범위에 포인트가 없습니다.")
            return
        # 레이어 전환
        self.viewer.add_layer("단면", section_pts, section_colors)
        self.viewer.set_layer_visible("정렬", False)
        self._chk_aligned.setChecked(False)
        self.viewer.set_layer_visible("원본", False)
        self._chk_original.setChecked(False)
        self.viewer.clear_polyline()
        # 현재 단면 정보 저장 (외곽 추출용)
        self._current_section = {
            'y_pos': y_pos, 'side_idx': side_idx,
            'fwd_idx': fwd_idx, 'up_idx': up_idx,
            'pts': section_pts,
        }
        # 이전 투영 모드 저장 후 정사 + Front View
        self._prev_perspective = self.viewer.perspective
        self.viewer.perspective = False
        self.viewer_panel._act_ortho.setChecked(True)
        self.viewer.set_front_view()
        self._section_active = True
        self._btn_close_section.setEnabled(True)
        self._btn_extract_contour.setEnabled(True)
        self._btn_measure.setEnabled(True)
        self.statusbar.showMessage(
            f"단면: Slice {row} (Y={y_pos:.2f}) | {len(section_pts):,} pts | 두께: {thickness}m"
        )
    def _extract_contour(self):
        """현재 단면의 외곽 추출."""
        if not self._section_active or self._current_section is None:
            return
        self._busy(True)
        try:
            self._extract_contour_impl()
        finally:
            self._busy(False)
    def _extract_contour_impl(self):
        sec = self._current_section
        method = self._contour_combo.currentData()
        method_label = self._contour_combo.currentText()
        pts_2d = sec['pts'][:, [sec['side_idx'], sec['up_idx']]]
        kwargs = self._get_contour_kwargs(method)
        try:
            contour_2d = PointCloud.extract_contour(pts_2d, method=method, **kwargs)
        except Exception as e:
            print(f"[Contour] {method} failed: {e}, fallback to radial")
            contour_2d = PointCloud.extract_contour(pts_2d, method='radial')
        if len(contour_2d) >= 3:
            contour_3d = np.zeros((len(contour_2d), 3))
            contour_3d[:, sec['side_idx']] = contour_2d[:, 0]
            contour_3d[:, sec['fwd_idx']] = sec['y_pos']
            contour_3d[:, sec['up_idx']] = contour_2d[:, 1]
            self.viewer.set_polyline(contour_3d)
            # segment 분석
            self._analyze_segments(contour_2d, pts_2d)
            self.statusbar.showMessage(
                f"외곽 추출 ({method_label}) | {len(contour_2d)} pts"
            )
        else:
            self.viewer.clear_polyline()
            self._seg_table.setRowCount(0)
            self.statusbar.showMessage("외곽 추출 실패: 포인트 부족")
    def _analyze_segments(self, contour_2d, pts_2d):
        """각 segment별 수직방향 검색범위 내 점군의 최대거리 계산."""
        n = len(contour_2d)
        threshold = self._spin_dist_threshold.value()
        search_range = self._spin_search_range.value()
        red_bg = QBrush(QColor(220, 60, 60))
        sec = self._current_section
        # 소팅 중 테이블 변경 방지
        self._seg_table.setSortingEnabled(False)
        self._seg_table.setRowCount(n)
        self._seg_analysis = []
        for i in range(n):
            p1 = contour_2d[i]
            p2 = contour_2d[(i + 1) % n]
            seg_vec = p2 - p1
            seg_len = np.linalg.norm(seg_vec)
            max_dist = 0.0
            max_pt_2d = None
            if seg_len >= 1e-10:
                normal = np.array([-seg_vec[1], seg_vec[0]]) / seg_len
                rel = pts_2d - p1
                proj_along = rel @ seg_vec / seg_len
                proj_perp = rel @ normal
                mask = ((proj_along >= 0) & (proj_along <= seg_len)
                        & (np.abs(proj_perp) <= search_range))
                if np.any(mask):
                    abs_perp = np.abs(proj_perp[mask])
                    max_idx_in_mask = np.argmax(abs_perp)
                    max_dist = abs_perp[max_idx_in_mask]
                    orig_indices = np.where(mask)[0]
                    max_pt_2d = pts_2d[orig_indices[max_idx_in_mask]]
            # 3D 좌표 복원
            max_pt_3d = None
            seg_p1_3d = np.zeros(3)
            seg_p2_3d = np.zeros(3)
            if sec is not None:
                for arr, pt2d in [(seg_p1_3d, p1), (seg_p2_3d, p2)]:
                    arr[sec['side_idx']] = pt2d[0]
                    arr[sec['fwd_idx']] = sec['y_pos']
                    arr[sec['up_idx']] = pt2d[1]
                if max_pt_2d is not None:
                    max_pt_3d = np.zeros(3)
                    max_pt_3d[sec['side_idx']] = max_pt_2d[0]
                    max_pt_3d[sec['fwd_idx']] = sec['y_pos']
                    max_pt_3d[sec['up_idx']] = max_pt_2d[1]
            self._seg_analysis.append({
                'idx': i, 'p1': p1, 'p2': p2, 'seg_len': seg_len,
                'max_dist': max_dist, 'max_pt_3d': max_pt_3d,
                'seg_p1_3d': seg_p1_3d, 'seg_p2_3d': seg_p2_3d,
            })
            # 테이블 채우기
            exceed = max_dist > threshold
            texts = [f"{i}", f"{seg_len:.3f}", f"{max_dist:.4f}"]
            for col, text in enumerate(texts):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, i)  # 원본 인덱스
                item.setTextAlignment(Qt.AlignCenter)
                if exceed:
                    item.setBackground(red_bg)
                self._seg_table.setItem(i, col, item)
        self._seg_table.setSortingEnabled(True)
        self._seg_table.resizeColumnsToContents()
    def _get_seg_from_row(self, row):
        """테이블 row에서 원본 segment 인덱스 반환 (소팅 대응)."""
        item = self._seg_table.item(row, 0)
        if item is None:
            return None
        orig_idx = item.data(Qt.UserRole)
        if orig_idx is not None and 0 <= orig_idx < len(self._seg_analysis):
            return self._seg_analysis[orig_idx]
        return None
    def _on_segment_dblclicked(self, row, col):
        """segment 더블클릭 → 확대 + segment 강조 + 최대거리 점 마커."""
        seg = self._get_seg_from_row(row)
        if seg is None or self._current_section is None:
            return
        sec = self._current_section
        p1_3d = seg['seg_p1_3d']
        p2_3d = seg['seg_p2_3d']
        # segment 중심으로 카메라 이동 + 확대
        mid_3d = (p1_3d + p2_3d) * 0.5
        seg_half = max(seg['seg_len'] * 0.5, 0.5)
        self.viewer._center = -mid_3d
        self.viewer._distance = seg_half * 4.0
        self.viewer._pan_screen = np.zeros(2)
        # segment 선분 강조 (폴리라인으로 표시)
        highlight = np.array([p1_3d, p2_3d], dtype=np.float32)
        self.viewer._highlight_seg = highlight
        # 최대거리 점 마커
        if seg['max_pt_3d'] is not None:
            self.viewer.set_marker(seg['max_pt_3d'], size=seg_half * 0.08)
        else:
            self.viewer.clear_marker()
        self.viewer.set_front_view()
        self.statusbar.showMessage(
            f"Segment {seg['idx']} | 길이: {seg['seg_len']:.3f}m | "
            f"최대거리: {seg['max_dist']:.4f}m"
        )
    def _close_section(self):
        self.viewer.remove_layer("단면")
        self.viewer.clear_polyline()
        self.viewer.clear_marker()
        self.viewer._highlight_seg = None
        self._seg_table.setRowCount(0)
        self._seg_analysis = []
        self.viewer.set_layer_visible("정렬", True)
        self._chk_aligned.setChecked(True)
        self._current_section = None
        if hasattr(self, '_prev_perspective'):
            self.viewer.perspective = self._prev_perspective
            if self._prev_perspective:
                self.viewer_panel._act_persp.setChecked(True)
            else:
                self.viewer_panel._act_ortho.setChecked(True)
        self._section_active = False
        self._btn_close_section.setEnabled(False)
        self._btn_extract_contour.setEnabled(False)
        self._btn_measure.setEnabled(False)
        self._btn_measure.setChecked(False)
        self._btn_measure_clear.setEnabled(False)
        self.viewer.set_measure_active(False)
        self.statusbar.showMessage("단면 닫기 → 정렬 뷰 복원")
    # ── 전체 외곽 면 생성 ──
    def _batch_contour_surface(self):
        """모든 slice에서 외곽 추출 → extrusion → 삼각형 메시 생성."""
        if not self._slices or self.pc_aligned is None:
            return
        conv = self.pc_aligned.convention
        fwd_idx = conv.forward_index
        side_idx = conv.side_index
        up_idx = conv.up_index
        pts = self.pc_aligned.points
        interval = self._spin_interval.value()
        half_interval = interval * 0.5
        thickness = self._spin_thickness.value()
        method = self._contour_combo.currentData()
        kwargs = self._get_contour_kwargs(method)
        total = len(self._slices)
        progress = QProgressDialog("외곽 면 생성 중...", "취소", 0, total, self)
        progress.setWindowTitle("전체 외곽 면 생성")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModal)
        all_verts = []
        all_norms = []
        for si, y_pos in enumerate(self._slices):
            progress.setValue(si)
            QApplication.processEvents()
            if progress.wasCanceled():
                break
            mask = np.abs(pts[:, fwd_idx] - y_pos) <= thickness * 0.5
            section_pts = pts[mask]
            if len(section_pts) < 3:
                continue
            pts_2d = section_pts[:, [side_idx, up_idx]]
            try:
                contour_2d = PointCloud.extract_contour(pts_2d, method=method, **kwargs)
            except Exception:
                contour_2d = PointCloud.extract_contour(pts_2d, method='radial')
            if len(contour_2d) < 3:
                continue
            y_front = y_pos - half_interval
            y_back = y_pos + half_interval
            n_pts = len(contour_2d)
            for j in range(n_pts):
                j_next = (j + 1) % n_pts
                p0 = np.zeros(3); p0[side_idx] = contour_2d[j, 0]; p0[fwd_idx] = y_front; p0[up_idx] = contour_2d[j, 1]
                p1 = np.zeros(3); p1[side_idx] = contour_2d[j_next, 0]; p1[fwd_idx] = y_front; p1[up_idx] = contour_2d[j_next, 1]
                p2 = np.zeros(3); p2[side_idx] = contour_2d[j, 0]; p2[fwd_idx] = y_back; p2[up_idx] = contour_2d[j, 1]
                p3 = np.zeros(3); p3[side_idx] = contour_2d[j_next, 0]; p3[fwd_idx] = y_back; p3[up_idx] = contour_2d[j_next, 1]
                edge1 = p1 - p0
                edge2 = p2 - p0
                normal = np.cross(edge1, edge2)
                n_len = np.linalg.norm(normal)
                if n_len > 1e-10:
                    normal /= n_len
                else:
                    normal = np.array([0, 0, 1], dtype=np.float64)
                all_verts.extend([p0, p1, p2, p1, p3, p2])
                all_norms.extend([normal] * 6)
        progress.setValue(total)
        if progress.wasCanceled():
            self.statusbar.showMessage("면 생성 취소됨")
            return
        if not all_verts:
            self.statusbar.showMessage("면 생성 실패: 유효한 외곽 없음")
            return
        verts = np.array(all_verts, dtype=np.float32)
        norms = np.array(all_norms, dtype=np.float32)
        self.viewer.set_mesh(verts, norms)
        n_tris = len(verts) // 3
        self.statusbar.showMessage(
            f"전체 외곽 면 생성 완료 | {total} slices | "
            f"{n_tris} triangles | 방식: {method}"
        )
