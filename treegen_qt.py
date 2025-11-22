# Minimal PyQt6 scaffold for treegen/pinegen UI
# Uses the existing tkinter script as a reference. This file is a starting point
# for porting; it creates main window, tabs, headers (title left, logo right),
# and a preview QLabel on the right � matching the layout of the tkinter version.

import sys
import os
import concurrent.futures
import multiprocessing
import random
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QTabWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QSlider, QProgressBar, QCheckBox, QFrame, QMessageBox, QSpinBox,
    QScrollArea, QListView, QMenuBar,
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QAction
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

# Reuse resource_path from the tkinter app for PyInstaller compatibility
import sys as _sys

from PIL.ImageQt import ImageQt

# Import core generation functions and export wrappers
try:
    from treegen_core import generate_treegen_preview, generate_pinegen_preview, export_tree, export_pine
    CORE_AVAILABLE = True
except Exception as e:
    print('Warning: could not import treegen_core:', e)
    CORE_AVAILABLE = False


def resource_path(filename):
    if hasattr(_sys, '_MEIPASS'):
        return os.path.join(_sys._MEIPASS, filename)
    return filename


class PreviewThread(QThread):
    finished = pyqtSignal(object, str)  # (PIL.Image, 'tree'|'pine')
    error = pyqtSignal(str)
    progress = pyqtSignal(float)  # 0.0 - 1.0

    def __init__(self, mode, params, palette_name, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.params = params
        self.palette_name = palette_name
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            def progress_cb(p):
                try:
                    self.progress.emit(max(0.0, min(1.0, float(p))))
                except Exception:
                    pass
            def cancel_check():
                return self._cancelled
            if self.mode == 'tree':
                img = generate_treegen_preview(self.params, self.palette_name, progress_callback=progress_cb, cancel_check=cancel_check)
            else:
                img = generate_pinegen_preview(self.params, self.palette_name, progress_callback=progress_cb, cancel_check=cancel_check)
            if self._cancelled:
                # treat as cancelled
                self.error.emit('Cancelled')
                return
            self.finished.emit(img, self.mode)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(str(e) + '\n' + tb)


class TreegenQtMain(QWidget):
    export_finished = pyqtSignal(str, str)  # mode, filename
    export_failed = pyqtSignal(str, str)    # mode, error

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voxel Tree Generator Studio (PyQt6)")
        self.resize(900, 900)

        if not CORE_AVAILABLE:
            print('Warning: core generation functions not available; preview and export disabled')

        layout = QVBoxLayout(self)

        # Add menu bar
        try:
            menubar = QMenuBar(self)
            layout.setMenuBar(menubar)
            file_menu = menubar.addMenu('File')
            close_action = QAction('Close', self)
            close_action.setShortcut('Ctrl+Q')
            close_action.triggered.connect(self.close)
            file_menu.addAction(close_action)

            # Help / About menu
            help_menu = menubar.addMenu('Help')
            about_action = QAction('About', self)
            about_action.setShortcut('F1')
            about_action.triggered.connect(self._show_about)
            help_menu.addAction(about_action)
        except Exception:
            pass

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        self.tree_tab = QWidget()
        self.pine_tab = QWidget()

        self.tabs.addTab(self.tree_tab, "Treegen")
        self.tabs.addTab(self.pine_tab, "Pinegen")

        # timers for debouncing preview updates
        self.tree_preview_timer = QTimer(self)
        self.tree_preview_timer.setSingleShot(True)
        self.tree_preview_timer.timeout.connect(self._start_tree_preview)

        self.pine_preview_timer = QTimer(self)
        self.pine_preview_timer.setSingleShot(True)
        self.pine_preview_timer.timeout.connect(self._start_pine_preview)

        # keep reference to running thread to avoid GC
        # track active preview threads (there can be overlapping ones)
        self._preview_threads = set()
        self._preview_thread = None

        # simple log file for diagnostics
        try:
            self._log_file = open(os.path.join(os.path.dirname(__file__), 'treegen_qt.log'), 'a', encoding='utf-8')
        except Exception:
            self._log_file = None

        # executor for export tasks (created on demand)
        self._export_executor = None

        # connect export signals
        self.export_finished.connect(self._on_export_finished)
        self.export_failed.connect(self._on_export_failed)

        self.build_tree_tab()
        self.build_pine_tab()

    def _make_header(self, parent_widget, title_text, logo_path, tint_color: str = None):
        """Create a professional header: logo centered above, title and subtitle below, all centered."""
        header = QFrame(parent_widget)
        header.setStyleSheet(
            """
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #232a34, stop:1 #1e2228);
            border-bottom: 2px solid #2980b9;
            """
        )
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 12, 0, 12)
        header_layout.setSpacing(6)

        # Logo(s) centered. Accept either a single path or an iterable of paths to place side-by-side.
        # If tint_color is provided, apply to any logo filename that contains 'text' to match theme.
        def _load_and_maybe_tint(pth):
            pm = QPixmap(resource_path(pth))
            if pm and not pm.isNull() and tint_color and 'text' in os.path.basename(pth).lower():
                try:
                    return self._tint_pixmap(pm, tint_color)
                except Exception:
                    return pm
            return pm

        if isinstance(logo_path, (list, tuple)):
            logos_widget = QWidget()
            logos_layout = QHBoxLayout(logos_widget)
            logos_layout.setContentsMargins(0, 0, 0, 0)
            logos_layout.setSpacing(8)
            for p in logo_path:
                lbl = QLabel()
                lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
                pix = _load_and_maybe_tint(p)
                if not pix.isNull():
                    lbl.setPixmap(pix.scaledToHeight(56, Qt.TransformationMode.SmoothTransformation))
                logos_layout.addWidget(lbl, 0, Qt.AlignmentFlag.AlignHCenter)
            header_layout.addWidget(logos_widget, 0, Qt.AlignmentFlag.AlignHCenter)
        else:
            logo = QLabel()
            logo.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            logo_pix = _load_and_maybe_tint(logo_path)
            if not logo_pix.isNull():
                logo.setPixmap(logo_pix.scaledToHeight(56, Qt.TransformationMode.SmoothTransformation))
            header_layout.addWidget(logo, 0, Qt.AlignmentFlag.AlignHCenter)

        # Title centered, bold, larger font
        title = QLabel(title_text)
        title.setStyleSheet("font-weight: 600; font-size: 18pt; color: #ecf0f1; letter-spacing: 1px;")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(title, 0, Qt.AlignmentFlag.AlignHCenter)

        # Optional subtitle/version (smaller, lighter)
        if "Treegen" in title_text:
            subtitle = QLabel("Voxel Tree Generator Studio (PyQt6)")
        else:
            subtitle = QLabel("Voxel Pine Generator Studio (PyQt6)")
        subtitle.setStyleSheet("font-size: 10pt; color: #b0b8c1; font-weight: 400;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(subtitle, 0, Qt.AlignmentFlag.AlignHCenter)

        return header

    def _tint_pixmap(self, pixmap: QPixmap, hex_color: str) -> QPixmap:
        """Return a tinted copy of `pixmap` using `hex_color` (e.g. '#6bb64f')."""
        try:
            img = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
            result = QPixmap(img.size())
            result.fill(Qt.GlobalColor.transparent)
            painter = QPainter(result)
            painter.drawPixmap(0, 0, pixmap)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(result.rect(), QColor(hex_color))
            painter.end()
            return result
        except Exception:
            return pixmap

    # helper to create a float slider mapped to a range
    def _create_float_slider(self, parent_layout, label_text, amin, amax, initial):
        parent_layout.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(1000)
        # map initial to slider
        t = int(round((initial - amin) / (amax - amin) * 1000)) if amax != amin else 0
        slider.setValue(max(0, min(1000, t)))
        parent_layout.addWidget(slider)
        return slider, (amin, amax)

    def build_tree_tab(self):
        tab = self.tree_tab
        main_layout = QVBoxLayout(tab)

        #header = self._make_header(tab, "Treegen v1.4.1 by NGNT", "treegen_brand.png")
        # Use two-part branding for the Treegen header: image + text side-by-side
        # Pass a tree tint color to colorize the text part of the header
        header = self._make_header(tab, "Treegen v1.5 by NGNT", ("img/treegen_image.png", "img/treegen_text.png"), tint_color='#6bb64f')
        # Override header styling for tree tab to use tree/forest colors
        try:
            header.setStyleSheet(
                """
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1f3a1e, stop:1 #0f2a12);
                border-bottom: 2px solid #4e9b3a;
                """
            )
        except Exception:
            pass
        main_layout.addWidget(header)

        # Content layout: left controls, right preview (preview anchored top-right)
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # Left controls panel (styled)
        controls_panel = QFrame()
        # Use earthy green background and subtle brown/green border for tree tab controls
        controls_panel.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #233a1f, stop:1 #162a12);"
            " border: 1px solid #3a5a2b; border-radius: 6px;"
        )
        controls_panel_layout = QVBoxLayout(controls_panel)
        controls_panel_layout.setContentsMargins(12, 12, 12, 12)
        controls_panel_layout.setSpacing(8)

        # header for controls panel
        panel_title = QLabel("Settings")
        panel_title.setStyleSheet("font-weight: 600; color: #ecf0f1;")
        controls_panel_layout.addWidget(panel_title)

        # Left controls placeholder
        controls_layout = QVBoxLayout()
        controls_frame_widget = QWidget()
        controls_frame_widget.setLayout(controls_layout)
        # Put the controls into a scroll area so the panel can be shorter than the controls list
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_frame_widget)
        controls_scroll.setStyleSheet("border: none; background: transparent;")
        controls_panel_layout.addWidget(controls_scroll)

        # Palette combo
        self.tree_palette_combo = QComboBox()
        tree_palette_dir = resource_path(os.path.join("palettes", "tree"))
        try:
            files = [f for f in os.listdir(tree_palette_dir) if f.endswith('.png')]
        except Exception:
            files = []
        if files:
            self.tree_palette_combo.addItems(files)
        controls_layout.addWidget(QLabel("Palette"))
        controls_layout.addWidget(self.tree_palette_combo)
        # Ensure the combo's popup is opaque and readable on tree theme
        try:
            self.tree_palette_combo.setView(QListView())
            self.tree_palette_combo.setStyleSheet(
                """
                QComboBox { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #6bb64f, stop:1 #2f7f2d); color: #ffffff; border: 1px solid #3a5a2b; border-radius: 4px; padding: 4px; }
                QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 20px; border-left: 1px solid #3a5a2b; }
                QComboBox QAbstractItemView { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #6bb64f, stop:1 #2f7f2d); color: #ffffff; selection-background-color: #7fcd61; selection-color: #ffffff; border: 1px solid #3a5a2b; }
                """
            )
        except Exception:
            pass

        # Size slider (0.1..3.0)
        self.tree_size_slider, self.tree_size_range = self._create_float_slider(controls_layout, "Size", 0.1, 3.0, 1.0)
        # Trunk size (0.1..1.2) - match treegen-pinegen.py upper bound
        self.tree_trunk_slider, self.tree_trunk_range = self._create_float_slider(controls_layout, "Trunk Size", 0.1, 1.2, 1.0)
        # Spread (0..1)
        self.tree_spread_slider, self.tree_spread_range = self._create_float_slider(controls_layout, "Spread", 0.0, 1.0, 0.5)
        # Twist (0..1)
        self.tree_twist_slider, self.tree_twist_range = self._create_float_slider(controls_layout, "Twist", 0.0, 1.0, 0.5)
        # Leafiness (0..3)
        self.tree_leaves_slider, self.tree_leaves_range = self._create_float_slider(controls_layout, "Leafiness", 0.0, 3.0, 1.0)
        # Gravity (-1..1)
        self.tree_gravity_slider, self.tree_gravity_range = self._create_float_slider(controls_layout, "Gravity", -1.0, 1.0, 0.0)
        # Wide (0..1)
        self.tree_wide_slider, self.tree_wide_range = self._create_float_slider(controls_layout, "Wide", 0.0, 1.0, 0.5)
        # Iterations (5..15) integer
        controls_layout.addWidget(QLabel("Iterations"))
        iterations_label = QLabel("Iterations")
        iterations_label.setStyleSheet("color: #ecf0f1; font-weight: 500;")
        controls_layout.addWidget(iterations_label)
        self.tree_iterations_spin = QSpinBox()
        self.tree_iterations_spin.setRange(5, 15)
        self.tree_iterations_spin.setValue(12)
        self.tree_iterations_spin.setStyleSheet(
            """
            QSpinBox { background: #233a1f; color: #eaf6e9; border: 1px solid #3a5a2b; border-radius: 4px; padding: 2px; }
            QSpinBox::up-button, QSpinBox::down-button { background: #3a5a2b; border: none, width: 12px; }
            QSpinBox::up-arrow, QSpinBox::down-arrow { width: 6px; height: 6px; }
            """
        )
        controls_layout.addWidget(self.tree_iterations_spin)
        # Seed (1..9999)
        controls_layout.addWidget(QLabel("Seed"))
        seed_label = QLabel("Seed")
        seed_label.setStyleSheet("color: #ecf0f1; font-weight: 500;")
        controls_layout.addWidget(seed_label)
        self.tree_seed_spin = QSpinBox()
        self.tree_seed_spin.setRange(1, 9999)
        self.tree_seed_spin.setValue(1)
        self.tree_seed_spin.setStyleSheet(
            """
            QSpinBox { background: #233a1f; color: #eaf6e9; border: 1px solid #3a5a2b; border-radius: 4px; padding: 2px; }
            QSpinBox::up-button, QSpinBox::down-button { background: #3a5a2b; border: none; width: 12px; }
            QSpinBox::up-arrow, QSpinBox::down-arrow { width: 6px; height: 6px; }
            """
        )
        controls_layout.addWidget(self.tree_seed_spin)

        # Generate button and progress bar (buttons moved to preview column)
        # keep placeholders for attribute creation later in preview section
        self.tree_progress = QProgressBar()
        self.tree_progress.setVisible(False)
        controls_layout.addWidget(self.tree_progress)

        # Preview progress
        self.tree_preview_progress = QProgressBar()
        self.tree_preview_progress.setVisible(False)
        self.tree_preview_progress.setRange(0, 100)
        controls_layout.addWidget(self.tree_preview_progress)

        controls_layout.addStretch(1)
        content_layout.addWidget(controls_panel, 1)

        # Right preview area
        preview_layout = QVBoxLayout()
        preview_frame_widget = QWidget()
        preview_frame_widget.setLayout(preview_layout)

        # Buttons above preview: grouped and styled inside a small panel
        btn_container = QFrame()
        btn_container.setStyleSheet("background: #172613; border: 1px solid #2f3b2a; border-radius: 6px;")
        btn_container_layout = QVBoxLayout(btn_container)
        btn_container_layout.setContentsMargins(8, 6, 8, 6)
        btn_container_layout.setSpacing(6)
        # Tree-themed button style: earthy green gradient matching tree tab theme
        _btn_style = (
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #6bb64f, stop:1 #2f7f2d);"
            " color: white; padding: 6px 12px; border: none; border-radius: 4px; font-weight: 600; }"
            "QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #7fcd61, stop:1 #3f8f3a); }"
            "QPushButton:disabled { background: #567154; color: #c9d9c9; }"
        )
        # top row (buttons)
        top_row = QHBoxLayout()
        top_row.addStretch(1)
        self.tree_randomize_btn = QPushButton("Randomize")
        self.tree_randomize_btn.setStyleSheet(_btn_style)
        self.tree_randomize_btn.clicked.connect(self._randomize_tree_sliders)
        top_row.addWidget(self.tree_randomize_btn)
        self.tree_generate_btn = QPushButton("Generate Tree")
        self.tree_generate_btn.setStyleSheet(_btn_style)
        self.tree_generate_btn.clicked.connect(self._on_tree_generate)
        top_row.addWidget(self.tree_generate_btn)
        top_row.addStretch(1)
        btn_container_layout.addLayout(top_row)
        # bottom row (checkbox) centered under buttons
        chk_row = QHBoxLayout()
        chk_row.addStretch(1)
        self.tree_open_after = QCheckBox("Open file after generation")
        self.tree_open_after.setChecked(True)
        # Styled checkbox so the indicator is visible on dark panels
        self.tree_open_after.setStyleSheet(
            """
            QCheckBox { color: #eaf6e9; spacing: 6px; }
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #4b6a3a; background: #172613; }
            QCheckBox::indicator:unchecked { background: #172613; }
            QCheckBox::indicator:checked { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #79c25d, stop:1 #3f8f3a); border: 1px solid #2f7f2d; }
            """
        )
        chk_row.addWidget(self.tree_open_after)
        chk_row.addStretch(1)
        btn_container_layout.addLayout(chk_row)
        preview_layout.addWidget(btn_container)

        # Center vertically and align preview to the right
        preview_layout.addStretch(1)

        self.tree_preview = QLabel()
        self.tree_preview.setFixedSize(320, 320)
        self.tree_preview.setStyleSheet("background: #0e1b0f; border: 1px solid #4e9b3a;")
        # align widget center-right within the preview column
        self.tree_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Dimension label shown above the preview
        self.tree_dim_label = QLabel("")
        self.tree_dim_label.setStyleSheet("color: #eaf6e9; font-weight: 500;")
        self.tree_dim_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.tree_dim_label)
        preview_layout.addWidget(self.tree_preview, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        preview_layout.addStretch(1)
        content_layout.addWidget(preview_frame_widget)

        # Leave preview empty until first generated preview appears
        self.tree_preview.setText("")

        # Connect controls to debounce timer
        controls = [
            self.tree_size_slider, self.tree_trunk_slider, self.tree_spread_slider, self.tree_twist_slider,
            self.tree_leaves_slider, self.tree_gravity_slider, self.tree_wide_slider,
            self.tree_iterations_spin, self.tree_seed_spin, self.tree_palette_combo
        ]
        for c in controls:
            if isinstance(c, QSlider):
                c.valueChanged.connect(lambda _: self.tree_preview_timer.start(300))
            elif isinstance(c, QSpinBox):
                c.valueChanged.connect(lambda _: self.tree_preview_timer.start(300))
            elif isinstance(c, QComboBox):
                c.currentIndexChanged.connect(lambda _: self.tree_preview_timer.start(300))

    def build_pine_tab(self):
        tab = self.pine_tab
        main_layout = QVBoxLayout(tab)

        #header = self._make_header(tab, "Treegen v1.4.1 by NGNT", "treegen_brand.png")
        # Use two-part branding for the Pinegen header: image + text side-by-side
        # Pass a pine tint color to colorize the text part of the header
        header = self._make_header(tab, "Pinegen v1.5 by NGNT", ("img/pinegen_image.png", "img/pinegen_text.png"), tint_color='#39a373')
        # Pine-themed header (deep pine greens with teal accent)
        try:
            header.setStyleSheet(
                """
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #143927, stop:1 #06261a);
                border-bottom: 2px solid #2f8f66;
                """
            )
        except Exception:
            pass
        main_layout.addWidget(header)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # Left controls panel (styled)
        controls_panel = QFrame()
        # Pine-themed controls panel: dark pine background with teal/bark border
        controls_panel.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #112f24, stop:1 #071b14);"
            " border: 1px solid #254c33; border-radius: 6px;"
        )
        controls_panel_layout = QVBoxLayout(controls_panel)
        controls_panel_layout.setContentsMargins(12, 12, 12, 12)
        controls_panel_layout.setSpacing(8)

        # header for controls panel
        panel_title = QLabel("Settings")
        panel_title.setStyleSheet("font-weight: 600; color: #ecf0f1;")
        controls_panel_layout.addWidget(panel_title)

        controls_layout = QVBoxLayout()
        controls_frame_widget = QWidget()
        controls_frame_widget.setLayout(controls_layout)
        # Put the controls into a scroll area so the panel can be shorter than the controls list
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_frame_widget)
        controls_scroll.setStyleSheet("border: none; background: transparent;")
        controls_panel_layout.addWidget(controls_scroll)

        self.pine_palette_combo = QComboBox()
        pine_palette_dir = resource_path(os.path.join("palettes", "pine"))
        try:
            files = [f for f in os.listdir(pine_palette_dir) if f.endswith('.png')]
        except Exception:
            files = []
        if files:
            self.pine_palette_combo.addItems(files)
        controls_layout.addWidget(QLabel("Palette"))
        controls_layout.addWidget(self.pine_palette_combo)
        # Ensure the combo's popup is opaque and readable on pine theme
        try:
            self.pine_palette_combo.setView(QListView())
            self.pine_palette_combo.setStyleSheet(
                """
                QComboBox { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #39a373, stop:1 #15764d); color: #ffffff; border: 1px solid #254c33; border-radius: 4px; padding: 4px; }
                QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 20px; border-left: 1px solid #254c33; }
                QComboBox QAbstractItemView { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #39a373, stop:1 #15764d); color: #ffffff; selection-background-color: #4fc389; selection-color: #ffffff; border: 1px solid #254c33; }
                """
            )
        except Exception:
            pass

        # Size
        self.pine_size_slider, self.pine_size_range = self._create_float_slider(controls_layout, "Size", 0.1, 3.0, 1.0)
        # Twist
        self.pine_twist_slider, self.pine_twist_range = self._create_float_slider(controls_layout, "Twist", 0.0, 4.0, 0.5)
        # Trunk Size
        self.pine_trunk_slider, self.pine_trunk_range = self._create_float_slider(controls_layout, "Trunk Size", 1.0, 3.0, 2.0)
        # Trunk Height
        self.pine_trunkheight_slider, self.pine_trunkheight_range = self._create_float_slider(controls_layout, "Trunk Height", 0.0, 5.0, 1.0)
        # Branch Density
        self.pine_branchdensity_slider, self.pine_branchdensity_range = self._create_float_slider(controls_layout, "Branch Density", 0.0, 3.0, 1.0)
        # Branch Length
        self.pine_branchlength_slider, self.pine_branchlength_range = self._create_float_slider(controls_layout, "Branch Length", 0.0, 3.0, 1.0)
        # Branch Direction
        self.pine_branchdir_slider, self.pine_branchdir_range = self._create_float_slider(controls_layout, "Branch Direction", -5.0, 5.0, -0.5)
        # Leafiness
        self.pine_leaves_slider, self.pine_leaves_range = self._create_float_slider(controls_layout, "Leafiness", 0.0, 2.0, 1.0)
        # Leaf Radius
        self.pine_leaf_radius_slider, self.pine_leaf_radius_range = self._create_float_slider(controls_layout, "Leaf Radius", 1.0, 4.0, 2.0)
        # Leaf Stretch
        self.pine_leaf_stretch_slider, self.pine_leaf_stretch_range = self._create_float_slider(controls_layout, "Leaf Stretch", 0.5, 3.0, 1.5)
        # Leaf Bias
        self.pine_leaf_bias_slider, self.pine_leaf_bias_range = self._create_float_slider(controls_layout, "Leaf Bias", -1.0, 1.0, -0.3)
        # Seed
        controls_layout.addWidget(QLabel("Seed"))
        seed_label = QLabel("Seed")
        seed_label.setStyleSheet("color: #ecf0f1; font-weight: 500;")
        controls_layout.addWidget(seed_label)
        self.pine_seed_spin = QSpinBox()
        self.pine_seed_spin.setRange(1, 9999)
        self.pine_seed_spin.setValue(1)
        self.pine_seed_spin.setStyleSheet(
            """
            QSpinBox { background: #0f3024; color: #e7f6ef; border: 1px solid #254c33; border-radius: 4px; padding: 2px; }
            QSpinBox::up-button, QSpinBox::down-button { background: #254c33; border: none; width: 12px; }
            QSpinBox::up-arrow, QSpinBox::down-arrow { width: 6px; height: 6px; }
            """
        )
        controls_layout.addWidget(self.pine_seed_spin)

        # Generate button and progress bar (buttons moved to preview column)
        # placeholders will be created in preview section
        self.pine_progress = QProgressBar()
        self.pine_progress.setVisible(False)
        controls_layout.addWidget(self.pine_progress)

        # Preview progress
        self.pine_preview_progress = QProgressBar()
        self.pine_preview_progress.setVisible(False)
        self.pine_preview_progress.setRange(0, 100)
        controls_layout.addWidget(self.pine_preview_progress)

        controls_layout.addStretch(1)
        content_layout.addWidget(controls_panel, 1)

        preview_layout = QVBoxLayout()
        preview_frame_widget = QWidget()
        preview_frame_widget.setLayout(preview_layout)

        # Buttons above preview: grouped and styled inside a small panel
        btn_container = QFrame()
        btn_container.setStyleSheet("background: #172613; border: 1px solid #2f3b2a; border-radius: 6px;")
        btn_container_layout = QVBoxLayout(btn_container)
        btn_container_layout.setContentsMargins(8, 6, 8, 6)
        btn_container_layout.setSpacing(6)
        # Pine-themed button style (teal/green gradient)
        _btn_style = (
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #39a373, stop:1 #15764d);"
            " color: white; padding: 6px 12px; border: none; border-radius: 4px; font-weight: 600; }"
            "QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #4fc389, stop:1 #1f8f5a); }"
            "QPushButton:disabled { background: #3c6b58; color: #cfefe0; }"
        )
        # top row (buttons)
        top_row = QHBoxLayout()
        top_row.addStretch(1)
        self.pine_randomize_btn = QPushButton("Randomize")
        self.pine_randomize_btn.setStyleSheet(_btn_style)
        self.pine_randomize_btn.clicked.connect(self._randomize_pine_sliders)
        top_row.addWidget(self.pine_randomize_btn)
        self.pine_generate_btn = QPushButton("Generate Pine")
        self.pine_generate_btn.setStyleSheet(_btn_style)
        self.pine_generate_btn.clicked.connect(self._on_pine_generate)
        top_row.addWidget(self.pine_generate_btn)
        top_row.addStretch(1)
        btn_container_layout.addLayout(top_row)
        # checkbox row under buttons
        chk_row = QHBoxLayout()
        chk_row.addStretch(1)
        self.pine_open_after = QCheckBox("Open file after generation")
        self.pine_open_after.setChecked(True)
        # Styled checkbox for visibility on dark panels
        self.pine_open_after.setStyleSheet(
            """
            QCheckBox { color: #e7f6ef; spacing: 6px; }
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #2f6a53; background: #081f18; }
            QCheckBox::indicator:unchecked { background: #081f18; }
            QCheckBox::indicator:checked { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #4fc389, stop:1 #1f8f5a); border: 1px solid #15764d; }
            """
        )
        chk_row.addWidget(self.pine_open_after)
        chk_row.addStretch(1)
        btn_container_layout.addLayout(chk_row)
        preview_layout.addWidget(btn_container)

        # Center vertically and align preview to the right
        preview_layout.addStretch(1)

        self.pine_preview = QLabel()
        self.pine_preview.setFixedSize(320, 320)
        self.pine_preview.setStyleSheet("background: #071913; border: 1px solid #2f8f66;")
        self.pine_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Dimension label shown above the preview
        self.pine_dim_label = QLabel("")
        self.pine_dim_label.setStyleSheet("color: #e7f6ef; font-weight: 500;")
        self.pine_dim_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.pine_dim_label)
        preview_layout.addWidget(self.pine_preview, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        preview_layout.addStretch(1)
        content_layout.addWidget(preview_frame_widget)

        # Leave preview empty until first generated preview appears
        self.pine_preview.setText("")

        # Connect controls to debounce timer
        controls = [
            self.pine_size_slider, self.pine_twist_slider, self.pine_trunk_slider, self.pine_trunkheight_slider,
            self.pine_branchdensity_slider, self.pine_branchlength_slider, self.pine_branchdir_slider,
            self.pine_leaves_slider, self.pine_leaf_radius_slider, self.pine_leaf_stretch_slider, self.pine_leaf_bias_slider,
            self.pine_seed_spin, self.pine_palette_combo
        ]
        for c in controls:
            if isinstance(c, QSlider):
                c.valueChanged.connect(lambda _: self.pine_preview_timer.start(300))
            elif isinstance(c, QSpinBox):
                c.valueChanged.connect(lambda _: self.pine_preview_timer.start(300))
            elif isinstance(c, QComboBox):
                c.currentIndexChanged.connect(lambda _: self.pine_preview_timer.start(300))

        # start initial preview
        self.pine_preview_timer.start(100)

    def _map_slider_to_range(self, val, amin, amax, bmin, bmax):
        # map val in [amin,amax] to [bmin,bmax]
        if amax == amin:
            return bmin
        t = (val - amin) / (amax - amin)
        return bmin + t * (bmax - bmin)

    def _start_tree_preview(self):
        if not CORE_AVAILABLE:
            return
        # assemble params from all controls
        def s2r(slider, r):
            v = slider.value()
            amin, amax = r
            return amin + (amax - amin) * (v / 1000.0)
        params = {
            'size': s2r(self.tree_size_slider, self.tree_size_range),
            'trunksize': s2r(self.tree_trunk_slider, self.tree_trunk_range),
            'spread': s2r(self.tree_spread_slider, self.tree_spread_range),
            'twisted': s2r(self.tree_twist_slider, self.tree_twist_range),
            'leaves': s2r(self.tree_leaves_slider, self.tree_leaves_range),
            'gravity': s2r(self.tree_gravity_slider, self.tree_gravity_range),
            'iterations': self.tree_iterations_spin.value(),
            'wide': s2r(self.tree_wide_slider, self.tree_wide_range),
            'seed': self.tree_seed_spin.value()
        }
        palette = self.tree_palette_combo.currentText() if self.tree_palette_combo.count() > 0 else ''
        # update estimated dimensions right away (estimator functions removed)
        try:
            # estimator removed; clear or keep existing label empty
            self.tree_dim_label.setText("")
        except Exception:
            pass
        # start thread (cancel any running)
        # cancel any running preview and start a new one; keep track of threads
        if self._preview_thread is not None and self._preview_thread.isRunning():
            try:
                self._preview_thread.cancel()
            except Exception:
                pass
        thr = PreviewThread('tree', params, palette, parent=self)
        self._register_preview_thread(thr)
        self._preview_thread = thr
        thr.progress.connect(self._on_tree_preview_progress)
        thr.finished.connect(self._on_preview_ready)
        thr.error.connect(self._on_preview_error)
        thr.finished.connect(lambda *_args, t=thr: self._unregister_preview_thread(t))
        thr.error.connect(lambda *_args, t=thr: self._unregister_preview_thread(t))
        # show preview progress
        self.tree_preview_progress.setVisible(True)
        self.tree_preview_progress.setValue(0)
        thr.start()

    def _start_pine_preview(self):
        if not CORE_AVAILABLE:
            return
        def s2r(slider, r):
            v = slider.value()
            amin, amax = r
            return amin + (amax - amin) * (v / 1000.0)
        params = {
            'size': s2r(self.pine_size_slider, self.pine_size_range),
            'twisted': s2r(self.pine_twist_slider, self.pine_twist_range),
            'trunksize': s2r(self.pine_trunk_slider, self.pine_trunk_range),
            'trunkheight': s2r(self.pine_trunkheight_slider, self.pine_trunkheight_range),
            'branchdensity': s2r(self.pine_branchdensity_slider, self.pine_branchdensity_range),
            'branchlength': s2r(self.pine_branchlength_slider, self.pine_branchlength_range),
            'branchdir': s2r(self.pine_branchdir_slider, self.pine_branchdir_range),
            'leaves': s2r(self.pine_leaves_slider, self.pine_leaves_range),
            'leaf_radius': s2r(self.pine_leaf_radius_slider, self.pine_leaf_radius_range),
            'leaf_stretch': s2r(self.pine_leaf_stretch_slider, self.pine_leaf_stretch_range),
            'leaf_bias': s2r(self.pine_leaf_bias_slider, self.pine_leaf_bias_range),
            'seed': self.pine_seed_spin.value()
        }
        palette = self.pine_palette_combo.currentText() if self.pine_palette_combo.count() > 0 else ''
        # estimator removed; clear pine dim label
        try:
            self.pine_dim_label.setText("")
        except Exception:
            pass
        if self._preview_thread is not None and self._preview_thread.isRunning():
            try:
                self._preview_thread.cancel()
            except Exception:
                pass
        thr = PreviewThread('pine', params, palette, parent=self)
        self._register_preview_thread(thr)
        self._preview_thread = thr
        thr.progress.connect(self._on_pine_preview_progress)
        thr.finished.connect(self._on_preview_ready)
        thr.error.connect(self._on_preview_error)
        thr.finished.connect(lambda *_args, t=thr: self._unregister_preview_thread(t))
        thr.error.connect(lambda *_args, t=thr: self._unregister_preview_thread(t))
        self.pine_preview_progress.setVisible(True)
        self.pine_preview_progress.setValue(0)
        thr.start()

    def _start_preview_thread(self, mode, params, palette_name):
        # legacy single-entrypoint kept for compatibility; not used when full wiring present
        if mode == 'tree':
            self._start_tree_preview()
        else:
            self._start_pine_preview()

    def _on_tree_preview_progress(self, p):
        try:
            self.tree_preview_progress.setValue(int(p * 100))
        except Exception:
            pass

    def _on_pine_preview_progress(self, p):
        try:
            self.pine_preview_progress.setValue(int(p * 100))
        except Exception:
            pass

    def _on_preview_ready(self, pil_image, mode):
        # Convert PIL Image to QPixmap and set on corresponding preview label
        try:
            qimage = ImageQt(pil_image).copy()
            pix = QPixmap.fromImage(QImage(qimage))
            if mode == 'tree':
                self.tree_preview.setPixmap(pix.scaled(self.tree_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                self.tree_preview_progress.setVisible(False)
                # update estimated dimensions from returned image size (approx voxels)
                try:
                    w, h = pil_image.size
                    # image corresponds to projection; keep z estimate from controls if available
                    # attempt to reuse existing params estimate
                    # keep existing label if set
                except Exception:
                    pass
            else:
                self.pine_preview.setPixmap(pix.scaled(self.pine_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                self.pine_preview_progress.setVisible(False)
        except Exception as e:
            print('Error converting preview image:', e)

    def _on_preview_error(self, msg):
        # Handle preview errors (cancellation is not an actual error)
        try:
            if msg and 'Cancelled' in msg:
                # hide progress bars
                try:
                    self.tree_preview_progress.setVisible(False)
                except Exception:
                    pass
                try:
                    self.pine_preview_progress.setVisible(False)
                except Exception:
                    pass
                return
            print('Preview generation error:', msg)
            try:
                QMessageBox.critical(self, 'Preview Error', f'Preview failed:\n{msg}')
            except Exception:
                pass
        except Exception:
            pass

    def _show_about(self):
        """Show About dialog with basic info."""
        try:
            about_text = (
                "Voxel Tree Generator Studio (PyQt6)\n"
                "Version 1.5\n\n"
                "Made by NGNT Creations\n"
                "Preview and export voxel trees and pines.\n"
                "https://github.com/NGNT/treegen-pinegen"
            )
            QMessageBox.about(self, 'About', about_text)
        except Exception:
            try:
                QMessageBox.information(self, 'About', 'Made by NGNT Creations')
            except Exception:
                pass

# ---------------- Export logic ----------------
    def _disable_tab_controls(self, tab_widget, disable=True):
        # disable or enable top-level tab contents (excluding preview QLabel)
        for child in tab_widget.findChildren(QWidget):
            # keep preview labels enabled so user sees the image
            if isinstance(child, QLabel) and (child is getattr(self, 'tree_preview', None) or child is getattr(self, 'pine_preview', None)):
                continue
            try:
                child.setDisabled(disable)
            except Exception:
                pass

    def _on_tree_generate(self):
        if not CORE_AVAILABLE:
            QMessageBox.critical(self, 'Error', 'Core generation not available')
            return
        size = self._map_slider_to_range(self.tree_size_slider.value(), 0, 1000, 0.1, 3.0)
        params = {
            'size': size,
            'twisted': self._map_slider_to_range(self.tree_twist_slider.value(), 0, 1000, 0.0, 1.0),
            'trunksize': self._map_slider_to_range(self.tree_trunk_slider.value(), 0, 1000, 0.1, 1.2),
            'trunkheight': 1.0,
            'branchdensity': 1.0,
            'branchlength': 1.0,
            'branchdir': -0.5,
            'leaves': self._map_slider_to_range(self.tree_leaves_slider.value(), 0.0, 1000, 0.0, 3.0),
            'leaf_radius': 2.0,
            'leaf_stretch': 1.5,
            'leaf_bias': -0.3,
            'seed': self.tree_seed_spin.value(),
            'iterations': self.tree_iterations_spin.value(),
            'spread': self._map_slider_to_range(self.tree_spread_slider.value(), 0, 1000, 0.0, 1.0),
            'gravity': self._map_slider_to_range(self.tree_gravity_slider.value(), 0, 1000, -1.0, 1.0),
            'wide': self._map_slider_to_range(self.tree_wide_slider.value(), 0, 1000, 0.0, 1.0)
        }
        palette = self.tree_palette_combo.currentText() if self.tree_palette_combo.count() > 0 else ''
        # prepare UI
        self.tree_progress.setVisible(True)
        self.tree_progress.setRange(0, 0)  # indeterminate
        self._disable_tab_controls(self.tree_tab, True)
        # start process pool executor if needed
        if self._export_executor is None:
            self._export_executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)
        future = self._export_executor.submit(export_tree, params, palette)
        # callback to handle completion
        future.add_done_callback(lambda f, m='tree': self._handle_export_future(f, m))

    def _on_pine_generate(self):
        if not CORE_AVAILABLE:
            QMessageBox.critical(self, 'Error', 'Core generation not available')
            return
        size = self._map_slider_to_range(self.pine_size_slider.value(), 0, 1000, 0.1, 3.0)
        params = {
            'size': size,
            'twisted': self._map_slider_to_range(self.pine_twist_slider.value(), 0, 1000, 0.0, 4.0),
            'trunksize': self._map_slider_to_range(self.pine_trunk_slider.value(), 0, 1000, 1.0, 3.0),
            'trunkheight': self._map_slider_to_range(self.pine_trunkheight_slider.value(), 0, 1000, 0.0, 5.0),
            'branchdensity': self._map_slider_to_range(self.pine_branchdensity_slider.value(), 0, 1000, 0.0, 3.0),
            'branchlength': self._map_slider_to_range(self.pine_branchlength_slider.value(), 0, 1000, 0.0, 3.0),
            'branchdir': self._map_slider_to_range(self.pine_branchdir_slider.value(), 0, 1000, -5.0, 5.0),
            'leaves': self._map_slider_to_range(self.pine_leaves_slider.value(), 0, 1000, 0.0, 2.0),
            'leaf_radius': self._map_slider_to_range(self.pine_leaf_radius_slider.value(), 0, 1000, 1.0, 4.0),
            'leaf_stretch': self._map_slider_to_range(self.pine_leaf_stretch_slider.value(), 0, 1000, 0.5, 3.0),
            'leaf_bias': self._map_slider_to_range(self.pine_leaf_bias_slider.value(), 0, 1000, -1.0, 1.0),
            'seed': self.pine_seed_spin.value()
        }
        palette = self.pine_palette_combo.currentText() if self.pine_palette_combo.count() > 0 else ''
        self.pine_progress.setVisible(True)
        self.pine_progress.setRange(0, 0)
        self._disable_tab_controls(self.pine_tab, True)
        if self._export_executor is None:
            self._export_executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)
        future = self._export_executor.submit(export_pine, params, palette)
        future.add_done_callback(lambda f, m='pine': self._handle_export_future(f, m))

    def _handle_export_future(self, future, mode):
        try:
            filename = future.result()
            # emit signal to main thread
            self.export_finished.emit(mode, filename)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.export_failed.emit(mode, str(e) + '\n' + tb)

    def _on_export_finished(self, mode, filename):
        # Re-enable UI, stop progress, show success and open file if requested
        if mode == 'tree':
            self.tree_progress.setVisible(False)
            self._disable_tab_controls(self.tree_tab, False)
            self.tree_generate_btn.setEnabled(True)
            # Display export result above the preview instead of a popup
            try:
                self.tree_dim_label.setText(f'Generated: {filename}')
            except Exception:
                pass
            if self.tree_open_after.isChecked():
                try:
                    if sys.platform.startswith('win'):
                        os.startfile(filename)
                    elif sys.platform == 'darwin':
                        import subprocess
                        subprocess.call(["open", filename])
                    else:
                        import subprocess
                        subprocess.call(["xdg-open", filename])
                except Exception:
                    pass
        else:
            self.pine_progress.setVisible(False)
            self._disable_tab_controls(self.pine_tab, False)
            self.pine_generate_btn.setEnabled(True)
            # Display export result above the preview instead of a popup
            try:
                self.pine_dim_label.setText(f'Generated: {filename}')
            except Exception:
                pass
            if self.pine_open_after.isChecked():
                try:
                    if sys.platform.startswith('win'):
                        os.startfile(filename)
                    elif sys.platform == 'darwin':
                        import subprocess
                        subprocess.call(["open", filename])
                    else:
                        import subprocess
                        subprocess.call(["xdg-open", filename])
                except Exception:
                    pass

    def _on_export_failed(self, mode, error_msg):
        if mode == 'tree':
            self.tree_progress.setVisible(False)
            self._disable_tab_controls(self.tree_tab, False)
            self.tree_generate_btn.setEnabled(True)
        else:
            self.pine_progress.setVisible(False)
            self._disable_tab_controls(self.pine_tab, False)
            self.pine_generate_btn.setEnabled(True)
        QMessageBox.critical(self, 'Export Failed', f'Export failed with error:\n{error_msg}')

    def cleanup(self):
        # Cancel running preview thread
        try:
            # cancel and wait for all registered preview threads
            for thr in list(self._preview_threads):
                try:
                    if thr.isRunning():
                        if self._log_file:
                            self._log_file.write(f'cleanup: cancelling thread {thr}\n')
                            self._log_file.flush()
                        try:
                            thr.cancel()
                        except Exception:
                            pass
                        try:
                            thr.requestInterruption()
                        except Exception:
                            pass
                except Exception:
                    pass
            # wait for them to stop
            for thr in list(self._preview_threads):
                try:
                    if not thr.wait(5000):
                        try:
                            thr.terminate()
                        except Exception:
                            pass
                        try:
                            thr.wait(2000)
                        except Exception:
                            pass
                except Exception:
                    pass
            self._preview_threads.clear()
            self._preview_thread = None
        except Exception:
            pass
        # Shutdown export executor
        try:
            if self._export_executor is not None:
                # try graceful shutdown first
                try:
                    self._export_executor.shutdown(wait=True, timeout=1)
                except TypeError:
                    # Python <3.9 doesn't support timeout param
                    try:
                        self._export_executor.shutdown(wait=False)
                    except Exception:
                        pass
                except Exception:
                    try:
                        self._export_executor.shutdown(wait=False)
                    except Exception:
                        pass
                self._export_executor = None
        except Exception:
            pass
        # flush log
        try:
            if self._log_file:
                self._log_file.write('cleanup done\n')
                self._log_file.flush()
                self._log_file.close()
        except Exception:
            pass

    def _register_preview_thread(self, thr: PreviewThread):
        try:
            self._preview_threads.add(thr)
            if self._log_file:
                self._log_file.write(f'registered preview thread {thr}\n')
                self._log_file.flush()
        except Exception:
            pass

    def _unregister_preview_thread(self, thr: PreviewThread):
        try:
            if thr in self._preview_threads:
                self._preview_threads.discard(thr)
            if self._log_file:
                self._log_file.write(f'unregistered preview thread {thr}\n')
                self._log_file.flush()
        except Exception:
            pass

    def _randomize_tree_sliders(self):
        # Randomize each slider/spin according to its defined range
        def set_slider_from_range(slider, r):
            amin, amax = r
            v = random.uniform(amin, amax)
            # map to 0..1000
            t = int(round((v - amin) / (amax - amin) * 1000)) if amax != amin else 0
            slider.setValue(max(0, min(1000, t)))

        set_slider_from_range(self.tree_size_slider, self.tree_size_range)
        set_slider_from_range(self.tree_trunk_slider, self.tree_trunk_range)
        set_slider_from_range(self.tree_spread_slider, self.tree_spread_range)
        set_slider_from_range(self.tree_twist_slider, self.tree_twist_range)
        set_slider_from_range(self.tree_leaves_slider, self.tree_leaves_range)
        set_slider_from_range(self.tree_gravity_slider, self.tree_gravity_range)
        # iterations spinbox
        self.tree_iterations_spin.setValue(random.randint(self.tree_iterations_spin.minimum(), self.tree_iterations_spin.maximum()))
        set_slider_from_range(self.tree_wide_slider, self.tree_wide_range)
        # seed
        self.tree_seed_spin.setValue(random.randint(self.tree_seed_spin.minimum(), self.tree_seed_spin.maximum()))
        # palette is not randomized here; set manually in UI
        # trigger preview
        self.tree_preview_timer.start(50)

    def _randomize_pine_sliders(self):
        def set_slider_from_range(slider, r):
            amin, amax = r
            v = random.uniform(amin, amax)
            t = int(round((v - amin) / (amax - amin) * 1000)) if amax != amin else 0
            slider.setValue(max(0, min(1000, t)))

        set_slider_from_range(self.pine_size_slider, self.pine_size_range)
        set_slider_from_range(self.pine_twist_slider, self.pine_twist_range)
        set_slider_from_range(self.pine_trunk_slider, self.pine_trunk_range)
        set_slider_from_range(self.pine_trunkheight_slider, self.pine_trunkheight_range)
        set_slider_from_range(self.pine_branchdensity_slider, self.pine_branchdensity_range)
        set_slider_from_range(self.pine_branchlength_slider, self.pine_branchlength_range)
        set_slider_from_range(self.pine_branchdir_slider, self.pine_branchdir_range)
        set_slider_from_range(self.pine_leaves_slider, self.pine_leaves_range)
        set_slider_from_range(self.pine_leaf_radius_slider, self.pine_leaf_radius_range)
        set_slider_from_range(self.pine_leaf_stretch_slider, self.pine_leaf_stretch_range)
        set_slider_from_range(self.pine_leaf_bias_slider, self.pine_leaf_bias_range)
        # seed
        self.pine_seed_spin.setValue(random.randint(self.pine_seed_spin.minimum(), self.pine_seed_spin.maximum()))
        # palette is not randomized here; set manually in UI
        # trigger preview
        self.pine_preview_timer.start(50)


def main():
    # On Windows when using multiprocessing in a frozen app, freeze_support is required
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass

    try:
        app = QApplication(sys.argv)
    except Exception as e:
        print("PyQt6 not installed or failed to initialize:", e)
        return

    w = TreegenQtMain()

    # Ensure we clean up threads and executors before exit to avoid QThread destruction while running
    def _cleanup():
        try:
            w.cleanup()
        except Exception:
            pass

    app.aboutToQuit.connect(_cleanup)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
