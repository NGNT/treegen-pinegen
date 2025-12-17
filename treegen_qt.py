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
from PIL import Image

# Add numpy import
import numpy as np

# Import core generation functions and export wrappers
from treegen_worker import generate_treegen_preview, generate_treegen_tree, export_tree
from pinegen_worker import generate_pinegen_preview, generate_pinegen_tree, export_pine
from birch_worker import generate_birchgen_preview, generate_birchgen_tree, export_birch

# Attempt to import palm worker
try:
    from palm_worker import generate_palm_preview, generate_palm_tree, export_palm
    PALM_AVAILABLE = True
except Exception:
    PALM_AVAILABLE = False

# Use internal palette manager for UI lists (avoid relying on PNG files)
try:
    import palette_worker
except Exception:
    palette_worker = None

# Fix birch script path detection to avoid calling resource_path before it's defined
birch_script = os.path.join(os.path.dirname(__file__) or '.', 'birch_worker.py')
BIRCH_AVAILABLE = os.path.exists(birch_script)

# Determine core availability (workers present)
try:
    # names imported earlier; if any missing, NameError will be raised
    _ = (generate_treegen_preview, generate_treegen_tree, export_tree,
         generate_pinegen_preview, generate_pinegen_tree, export_pine)
    CORE_AVAILABLE = True
except NameError:
    CORE_AVAILABLE = False

# Determine birch availability: set based on whether birch worker functions imported
try:
    _ = (generate_birchgen_preview, generate_birchgen_tree, export_birch)
    BIRCH_AVAILABLE = True
except NameError:
    BIRCH_AVAILABLE = False

# Provide a simple local birch_preset for UI use (keeps UI functional without importing functions)
def birch_preset(seed=None):
    if seed is None:
        seed = random.randint(1, 9999)
    return {
        'seed': int(seed),
        'size': 1.1,
        'trunksize': 0.35,
        'spread': 0.5,
        'twisted': 0.1,
        'leaves': 1.0,
        'gravity': 0.25,
        'iterations': 6,
        'wide': 0.1,
    }


def resource_path(filename):
    if hasattr(_sys, '_MEIPASS'):
        return os.path.join(_sys._MEIPASS, filename)
    return filename


# Add project_voxels_to_image function here for use in main process
def project_voxels_to_image(voxels, palette, grid_size, view='side'):
    if view == 'top':
        proj = voxels.max(axis=2)
    elif view == 'front':
        proj = voxels.max(axis=1).T[::-1, :]
    else:
        proj = voxels.max(axis=0).T[::-1, :]
    img_arr = np.zeros((grid_size, grid_size, 4), np.uint8)
    for idx, rgba in enumerate(palette):
        arr = np.array(rgba, dtype=np.uint8)
        img_arr[proj == idx] = arr
    return Image.fromarray(img_arr, 'RGBA')


class PreviewThread(QThread):
    finished = pyqtSignal(object, str)  # (PIL.Image, 'tree'|'pine'|'birch')
    error = pyqtSignal(str)
    progress = pyqtSignal(float)  # 0.0 - 1.0

    def __init__(self, mode, params, palette_name, executor, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.params = params
        self.palette_name = palette_name
        self.executor = executor
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            # Submit generation to process pool
            if self.mode == 'tree':
                future = self.executor.submit(generate_treegen_tree, self.params, self.palette_name, grid_size=256, preview=False)
            elif self.mode == 'pine':
                future = self.executor.submit(generate_pinegen_tree, self.params, self.palette_name, grid_size=256, preview=False)
            elif self.mode == 'birch':
                future = self.executor.submit(generate_birchgen_tree, self.params, self.palette_name, grid_size=256, preview=False)
            elif self.mode == 'palm':
                future = self.executor.submit(generate_palm_tree, self.params, self.palette_name, grid_size=256, preview=False)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            # Wait for result, checking for cancellation
            while not future.done():
                if self._cancelled:
                    future.cancel()
                    self.error.emit('Cancelled')
                    return
                self.msleep(50)  # poll every 50ms

            voxels, palette = future.result()

            # Project in main process
            if self.mode == 'palm':
                # Swap Y/Z for palm preview to match correct orientation
                voxels = np.swapaxes(voxels, 1, 2)
            img_full = project_voxels_to_image(voxels, palette, 256, view='front')
            img = img_full.resize((192, 192), Image.NEAREST)  # 64*3=192

            if self._cancelled:
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
        self.birch_tab = QWidget()
        self.palm_tab = QWidget()

        self.tabs.addTab(self.tree_tab, "Treegen")
        self.tabs.addTab(self.pine_tab, "Pinegen")
        self.tabs.addTab(self.birch_tab, "Birchgen")
        self.tabs.addTab(self.palm_tab, "Palmgen")

        # timers for debouncing preview updates
        self.tree_preview_timer = QTimer(self)
        self.tree_preview_timer.setSingleShot(True)
        self.tree_preview_timer.timeout.connect(self._start_tree_preview)

        self.pine_preview_timer = QTimer(self)
        self.pine_preview_timer.setSingleShot(True)
        self.pine_preview_timer.timeout.connect(self._start_pine_preview)

        self.birch_preview_timer = QTimer(self)
        self.birch_preview_timer.setSingleShot(True)
        self.birch_preview_timer.timeout.connect(self._start_birch_preview)

        self.palm_preview_timer = QTimer(self)
        self.palm_preview_timer.setSingleShot(True)
        self.palm_preview_timer.timeout.connect(self._start_palm_preview)

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
        self._export_executor = concurrent.futures.ProcessPoolExecutor(max_workers=6)

        # connect export signals
        self.export_finished.connect(self._on_export_finished)
        self.export_failed.connect(self._on_export_failed)

        self.build_tree_tab()
        self.build_pine_tab()
        self.build_birch_tab()
        self.build_palm_tab()

        # Connect tab change to trigger preview
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Start initial previews for all tabs
        self.tree_preview_timer.start(100)
        self.pine_preview_timer.start(100)
        self.birch_preview_timer.start(100)
        self.palm_preview_timer.start(100)

    def _map_slider_to_range(self, val, amin, amax, bmin, bmax):
        """Map value `val` in [amin, amax] to range [bmin, bmax].
        Returns bmin when amax == amin to avoid division by zero.
        """
        try:
            if amax == amin:
                return bmin
            t = (float(val) - float(amin)) / float(amax - amin)
            return bmin + t * (bmax - bmin)
        except Exception:
            return bmin

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
        header = self._make_header(tab, "Treegen by NGNT", ("img/treegen_image.png", "img/treegen_text.png"), tint_color='#6bb64f')
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
        # Populate from internal palette manager if available
        try:
            if palette_worker is not None and hasattr(palette_worker, '_palette_manager'):
                all_files = palette_worker._palette_manager.list_palettes()
                # Filter to tree-related palettes
                tree_palettes = [p for p in all_files if p in ['Basic', 'Default Tree', 'Sapling', 'Oak Variant 1', 'Oak Variant 2', 'Autumn 1', 'Autumn 2', 'Blossom, Cherry', 'Blossom, Apple', 'Dead 1', 'Dead 2']]
            else:
                tree_palette_dir = resource_path(os.path.join("palettes", "tree"))
                tree_palettes = [f for f in os.listdir(tree_palette_dir) if f.endswith('.png')]
        except Exception:
            tree_palettes = []
        if tree_palettes:
            # internal-only list (preferred) - show palette keys as-is
            self.tree_palette_combo.addItems(tree_palettes)

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

        # Apply tree-themed slider styles
        tree_slider_style = """
        QSlider { background: transparent; }
        QSlider::groove:horizontal { height: 8px; background: #3a5a2b; border-radius: 4px; }
        QSlider::handle:horizontal { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #6bb64f, stop:1 #2f7f2d); border: 1px solid #2f7f2d; width: 18px; margin: -5px 0; border-radius: 9px; }
        QSlider::handle:horizontal:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #7fcd61, stop:1 #3f8f3a); }
        """
        for slider in [self.tree_size_slider, self.tree_trunk_slider, self.tree_spread_slider, self.tree_twist_slider, self.tree_leaves_slider, self.tree_gravity_slider, self.tree_wide_slider]:
            slider.setStyleSheet(tree_slider_style)

        # Iterations (5..15) integer
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
        # Seed as slider (1..9999)
        self.tree_seed_slider, self.tree_seed_range = self._create_float_slider(controls_layout, "Seed", 1, 9999, 1)
        # apply the same tree slider style used earlier
        self.tree_seed_slider.setStyleSheet(tree_slider_style)

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
            self.tree_iterations_spin, self.tree_seed_slider, self.tree_palette_combo
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
        header = self._make_header(tab, "Pinegen by NGNT", ("img/pinegen_image.png", "img/pinegen_text.png"), tint_color='#39a373')
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
        # Populate from internal palette manager if available
        try:
            if palette_worker is not None and hasattr(palette_worker, '_palette_manager'):
                all_files = palette_worker._palette_manager.list_palettes()
                # Filter to pine-related palettes
                pine_palettes = [p for p in all_files if p in ['Default Pine', 'Basic Pine', 'Red Pine', 'Pine Sapling', 'Scots Pine']]
            else:
                pine_palette_dir = resource_path(os.path.join("palettes", "pine"))
                pine_palettes = [f for f in os.listdir(pine_palette_dir) if f.endswith('.png')]
        except Exception:
            pine_palettes = []
        if pine_palettes:
            self.pine_palette_combo.addItems(pine_palettes)
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

        # Apply pine-themed slider styles
        pine_slider_style = """
        QSlider { background: transparent; }
        QSlider::groove:horizontal { height: 8px; background: #254c33; border-radius: 4px; }
        QSlider::handle:horizontal { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #39a373, stop:1 #15764d); border: 1px solid #15764d; width: 18px; margin: -5px 0; border-radius: 9px; }
        QSlider::handle:horizontal:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #4fc389, stop:1 #1f8f5a); }
        """
        for slider in [self.pine_size_slider, self.pine_twist_slider, self.pine_trunk_slider, self.pine_trunkheight_slider, self.pine_branchdensity_slider, self.pine_branchlength_slider, self.pine_branchdir_slider, self.pine_leaves_slider, self.pine_leaf_radius_slider, self.pine_leaf_stretch_slider, self.pine_leaf_bias_slider]:
            slider.setStyleSheet(pine_slider_style)

        # Seed (slider 1..9999)
        self.pine_seed_slider, self.pine_seed_range = self._create_float_slider(controls_layout, "Seed", 1, 9999, 1)
        # apply same pine slider style
        self.pine_seed_slider.setStyleSheet(pine_slider_style)

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
            self.pine_seed_slider, self.pine_palette_combo
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

    def build_birch_tab(self):
        tab = self.birch_tab
        main_layout = QVBoxLayout(tab)

        # Header for Birchgen
        header = self._make_header(tab, "Birchgen by NGNT", ("img/birchgen_image.png", "img/birch_text.png"), tint_color='#8B4513')  # Brown tint for birch
        # Birch-themed header (browns and whites)
        try:
            header.setStyleSheet(
                """
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #D2B48C, stop:1 #8B4513);
                border-bottom: 2px solid #A0522D;
                """
            )
        except Exception:
            pass
        main_layout.addWidget(header)

        # Content layout: left controls, right preview
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # Left controls panel
        controls_panel = QFrame()
        controls_panel.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #F5DEB3, stop:1 #D2B48C);"
            " border: 1px solid #A0522D; border-radius: 6px;"
        )
        controls_panel_layout = QVBoxLayout(controls_panel)
        controls_panel_layout.setContentsMargins(12, 12, 12, 12)
        controls_panel_layout.setSpacing(8)

        panel_title = QLabel("Settings")
        panel_title.setStyleSheet("font-weight: 600; color: #8B4513;")
        controls_panel_layout.addWidget(panel_title)

        controls_layout = QVBoxLayout()
        controls_frame_widget = QWidget()
        controls_frame_widget.setLayout(controls_layout)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_frame_widget)
        controls_scroll.setStyleSheet("border: none; background: transparent;")
        controls_panel_layout.addWidget(controls_scroll)

        # Palette combo for birch (use tree palettes for now)
        self.birch_palette_combo = QComboBox()
        try:
            if palette_worker is not None and hasattr(palette_worker, '_palette_manager'):
                all_files = palette_worker._palette_manager.list_palettes()
                # Filter to birch-related palettes
                birch_palettes = [p for p in all_files if p in ['Birch Variant 1', 'White Birch', 'Yellow Birch', 'Gray Birch', 'Birch Variant 2']]
            else:
                birch_palette_dir = resource_path(os.path.join("palettes", "tree"))
                birch_palettes = [f for f in os.listdir(birch_palette_dir) if f.endswith('.png')]
        except Exception:
            birch_palettes = []
        if birch_palettes:
            self.birch_palette_combo.addItems(birch_palettes)
        controls_layout.addWidget(QLabel("Palette"))
        controls_layout.addWidget(self.birch_palette_combo)
        try:
            self.birch_palette_combo.setView(QListView())
            self.birch_palette_combo.setStyleSheet(
                """
                QComboBox { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #D2B48C, stop:1 #8B4513); color: #ffffff; border: 1px solid #A0522D; border-radius: 4px; padding: 4px; }
                QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 20px; border-left: 1px solid #A0522D; }
                QComboBox QAbstractItemView { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #D2B48C, stop:1 #8B4513); color: #ffffff; selection-background-color: #F5DEB3; selection-color: #8B4513; border: 1px solid #A0522D; }
                """
            )
        except Exception:
            pass

        # Sliders for birch parameters
        self.birch_size_slider, self.birch_size_range = self._create_float_slider(controls_layout, "Size", 0.1, 3.0, 1.1)
        # Birch trunks are typically slender; lower range and default to keep trunks thin
        self.birch_trunk_slider, self.birch_trunk_range = self._create_float_slider(controls_layout, "Trunk Size", 0.05, 0.8, 0.28)
        self.birch_spread_slider, self.birch_spread_range = self._create_float_slider(controls_layout, "Spread", 0.1, 0.4, 0.24)
        self.birch_twist_slider, self.birch_twist_range = self._create_float_slider(controls_layout, "Twist", 0.0, 1.0, 0.1)
        self.birch_leaves_slider, self.birch_leaves_range = self._create_float_slider(controls_layout, "Leafiness", 0.0, 3.0, 1.0)
        self.birch_gravity_slider, self.birch_gravity_range = self._create_float_slider(controls_layout, "Gravity", -1.0, 1.0, 0.25)
        self.birch_wide_slider, self.birch_wide_range = self._create_float_slider(controls_layout, "Wide", 0.05, 0.2, 0.1)

        # Apply birch-themed slider styles
        birch_slider_style = """
        QSlider { background: transparent; }
        QSlider::groove:horizontal { height: 8px; background: #A0522D; border-radius: 4px; }
        QSlider::handle:horizontal { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #D2B48C, stop:1 #8B4513); border: 1px solid #8B4513; width: 18px; margin: -5px 0; border-radius: 9px; }
        QSlider::handle:horizontal:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #F5DEB3, stop:1 #A0522D); }
        """
        for slider in [self.birch_size_slider, self.birch_trunk_slider, self.birch_spread_slider, self.birch_twist_slider, self.birch_leaves_slider, self.birch_gravity_slider, self.birch_wide_slider]:
            slider.setStyleSheet(birch_slider_style)

        controls_layout.addWidget(QLabel("Iterations"))
        iterations_label = QLabel("Iterations")
        iterations_label.setStyleSheet("color: #8B4513; font-weight: 500;")
        controls_layout.addWidget(iterations_label)
        self.birch_iterations_spin = QSpinBox()
        self.birch_iterations_spin.setRange(5, 14)
        self.birch_iterations_spin.setValue(8)
        self.birch_iterations_spin.setStyleSheet(
            """
            QSpinBox { background: #F5DEB3; color: #8B4513; border: 1px solid #A0522D; border-radius: 4px; padding: 2px; }
            QSpinBox::up-button, QSpinBox::down-button { background: #A0522D; border: none; width: 12px; }
            QSpinBox::up-arrow, QSpinBox::down-arrow { width: 6px; height: 6px; }
        """
        )
        controls_layout.addWidget(self.birch_iterations_spin)

        # Seed (slider 1..9999)
        self.birch_seed_slider, self.birch_seed_range = self._create_float_slider(controls_layout, "Seed", 1, 9999, 1)
        # apply birch slider style already defined above
        self.birch_seed_slider.setStyleSheet(birch_slider_style)

        self.birch_progress = QProgressBar()
        self.birch_progress.setVisible(False)
        controls_layout.addWidget(self.birch_progress)

        self.birch_preview_progress = QProgressBar()
        self.birch_preview_progress.setVisible(False)
        self.birch_preview_progress.setRange(0, 100)
        controls_layout.addWidget(self.birch_preview_progress)

        controls_layout.addStretch(1)
        content_layout.addWidget(controls_panel, 1)

        # Right preview area
        preview_layout = QVBoxLayout()
        preview_frame_widget = QWidget()
        preview_frame_widget.setLayout(preview_layout)

        btn_container = QFrame()
        btn_container.setStyleSheet("background: #D2B48C; border: 1px solid #A0522D; border-radius: 6px;")
        btn_container_layout = QVBoxLayout(btn_container)
        btn_container_layout.setContentsMargins(8, 6, 8, 6)
        btn_container_layout.setSpacing(6)

        _btn_style = (
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #D2B48C, stop:1 #8B4513);"
            " color: white; padding: 6px 12px; border: none; border-radius: 4px; font-weight: 600; }"
            "QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #F5DEB3, stop:1 #A0522D); }"
            "QPushButton:disabled { background: #A0522D; color: #F5DEB3; }"
        )

        # top row (buttons)
        top_row = QHBoxLayout()
        top_row.addStretch(1)
        self.birch_randomize_btn = QPushButton("Randomize")
        self.birch_randomize_btn.setStyleSheet(_btn_style)
        self.birch_randomize_btn.clicked.connect(self._randomize_birch_sliders)
        top_row.addWidget(self.birch_randomize_btn)
        self.birch_generate_btn = QPushButton("Generate Birch")
        self.birch_generate_btn.setStyleSheet(_btn_style)
        self.birch_generate_btn.clicked.connect(self._on_birch_generate)
        top_row.addWidget(self.birch_generate_btn)
        top_row.addStretch(1)
        btn_container_layout.addLayout(top_row)
        # checkbox row under buttons
        chk_row = QHBoxLayout()
        chk_row.addStretch(1)
        self.birch_open_after = QCheckBox("Open file after generation")
        self.birch_open_after.setChecked(True)
        # Styled checkbox for visibility on dark panels
        self.birch_open_after.setStyleSheet(
            """
            QCheckBox { color: #e7f6ef; spacing: 6px; }
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #8a6a3a; background: #0f0f0f; }
            QCheckBox::indicator:unchecked { background: #0f0f0f; }
            QCheckBox::indicator:checked { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #f3c500, stop:1 #ab7e26); border: 1px solid #9f7d36; }
            """
        )
        chk_row.addWidget(self.birch_open_after)
        chk_row.addStretch(1)
        btn_container_layout.addLayout(chk_row)
        preview_layout.addWidget(btn_container)

        # Center vertically and align preview to the right
        preview_layout.addStretch(1)

        self.birch_preview = QLabel()
        self.birch_preview.setFixedSize(320, 320)
        self.birch_preview.setStyleSheet("background: #F5F5DC; border: 1px solid #A0522D;")
        self.birch_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.birch_dim_label = QLabel("")
        self.birch_dim_label.setStyleSheet("color: #8B4513; font-weight: 500;")
        self.birch_dim_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.birch_dim_label)
        preview_layout.addWidget(self.birch_preview, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        preview_layout.addStretch(1)
        content_layout.addWidget(preview_frame_widget)

        # Leave preview empty until first generated preview appears
        self.birch_preview.setText("")

        # Connect controls to debounce timer
        controls = [
            self.birch_size_slider, self.birch_trunk_slider, self.birch_spread_slider, self.birch_twist_slider,
            self.birch_leaves_slider, self.birch_gravity_slider, self.birch_wide_slider,
            self.birch_iterations_spin, self.birch_seed_slider, self.birch_palette_combo
        ]
        for c in controls:
            if isinstance(c, QSlider):
                c.valueChanged.connect(lambda _: self.birch_preview_timer.start(300))
            elif isinstance(c, QSpinBox):
                c.valueChanged.connect(lambda _: self.birch_preview_timer.start(300))
            elif isinstance(c, QComboBox):
                c.currentIndexChanged.connect(lambda _: self.birch_preview_timer.start(300))

        # start initial preview
        self.birch_preview_timer.start(100)

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
            'seed': int(s2r(self.tree_seed_slider, self.tree_seed_range))
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
        thr = PreviewThread('tree', params, palette, self._export_executor, parent=self)
        self._register_preview_thread(thr)
        self._preview_thread = thr
        thr.progress.connect(self._on_tree_preview_progress)
        thr.finished.connect(self._on_preview_ready)
        thr.error.connect(self._on_preview_error)
        thr.finished.connect(lambda *_args, t=thr: self._unregister_preview_thread(t))
        thr.error.connect(lambda *_args, t=thr: self._unregister_preview_thread(t))
        # show preview progress as indeterminate since no progress from subprocess
        self.tree_preview_progress.setRange(0, 0)
        self.tree_preview_progress.setVisible(True)
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
            'seed': int(s2r(self.pine_seed_slider, self.pine_seed_range))
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
        thr = PreviewThread('pine', params, palette, self._export_executor, parent=self)
        self._register_preview_thread(thr)
        self._preview_thread = thr
        thr.progress.connect(self._on_pine_preview_progress)
        thr.finished.connect(self._on_preview_ready)
        thr.error.connect(self._on_preview_error)
        thr.finished.connect(lambda *_args, t=thr: self._unregister_preview_thread(t))
        thr.error.connect(lambda *_args, t=thr: self._unregister_preview_thread(t))
        self.pine_preview_progress.setRange(0, 0)
        self.pine_preview_progress.setVisible(True)
        thr.start()

    def _start_birch_preview(self):
        if not BIRCH_AVAILABLE:
            return
        def s2r(slider, r):
            v = slider.value()
            amin, amax = r
            return amin + (amax - amin) * (v / 1000.0)
        params = {
            'size': s2r(self.birch_size_slider, self.birch_size_range),
            'trunksize': s2r(self.birch_trunk_slider, self.birch_trunk_range),
            'spread': s2r(self.birch_spread_slider, self.birch_spread_range),
            'twisted': s2r(self.birch_twist_slider, self.birch_twist_range),
            'leaves': s2r(self.birch_leaves_slider, self.birch_leaves_range),
            'gravity': s2r(self.birch_gravity_slider, self.birch_gravity_range),
            'iterations': self.birch_iterations_spin.value(),
            'wide': s2r(self.birch_wide_slider, self.birch_wide_range),
            'seed': int(s2r(self.birch_seed_slider, self.birch_seed_range))
        }
        palette = self.birch_palette_combo.currentText() if self.birch_palette_combo.count() > 0 else ''
        try:
            self.birch_dim_label.setText("")
        except Exception:
            pass
        if self._preview_thread is not None and self._preview_thread.isRunning():
            try:
                self._preview_thread.cancel()
            except Exception:
                pass
        thr = PreviewThread('birch', params, palette, self._export_executor, parent=self)
        self._register_preview_thread(thr)
        self._preview_thread = thr
        thr.progress.connect(self._on_birch_preview_progress)
        thr.finished.connect(self._on_preview_ready)
        thr.error.connect(self._on_preview_error)
        thr.finished.connect(lambda *_args, t=thr: self._unregister_preview_thread(t))
        thr.error.connect(lambda *_args, t=thr: self._unregister_preview_thread(t))
        self.birch_preview_progress.setRange(0, 0)
        self.birch_preview_progress.setVisible(True)
        thr.start()
    
    def build_palm_tab(self):
        tab = self.palm_tab
        main_layout = QVBoxLayout(tab)
        header = self._make_header(tab, "Palmgen by NGNT", ("img/palm_image.png", "img/palmgen_text.png"), tint_color='#c27c2a')
        try:
            header.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2e2b19, stop:1 #1b1208); border-bottom: 2px solid #b06a1f;")
        except Exception:
            pass
        main_layout.addWidget(header)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        controls_panel = QFrame()
        controls_panel.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2b2412, stop:1 #1a1207); border: 1px solid #5a3e21; border-radius: 6px;")
        controls_panel_layout = QVBoxLayout(controls_panel)
        controls_panel_layout.setContentsMargins(12,12,12,12)
        controls_panel_layout.setSpacing(8)

        panel_title = QLabel("Palm Settings")
        panel_title.setStyleSheet("font-weight: 600; color: #e9d8c1;")
        controls_panel_layout.addWidget(panel_title)

        controls_layout = QVBoxLayout()
        controls_frame_widget = QWidget()
        controls_frame_widget.setLayout(controls_layout)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_frame_widget)
        controls_scroll.setStyleSheet("border: none; background: transparent;")
        controls_panel_layout.addWidget(controls_scroll)

        # Palette combo
        self.palm_palette_combo = QComboBox()
        try:
            if palette_worker is not None and hasattr(palette_worker, '_palette_manager'):
                all_files = palette_worker._palette_manager.list_palettes()
                palm_palettes = [p for p in all_files if 'palm' in p.lower() or 'tropical' in p.lower()]
            else:
                palm_palette_dir = resource_path(os.path.join("palettes","palm"))
                palm_palettes = [f for f in os.listdir(palm_palette_dir) if f.endswith('.png')]
        except Exception:
            palm_palettes = []
        if palm_palettes:
            self.palm_palette_combo.addItems(palm_palettes)
        controls_layout.addWidget(QLabel("Palette"))
        controls_layout.addWidget(self.palm_palette_combo)
        # Make the combo's popup opaque/readable and match palm theme
        try:
            self.palm_palette_combo.setView(QListView())
            self.palm_palette_combo.setStyleSheet(
                """
                QComboBox { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #c27c2a, stop:1 #b06a1f); color: #ffffff; border: 1px solid #6b4523; border-radius: 4px; padding: 4px; }
                QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 20px; border-left: 1px solid #6b4523; }
                QComboBox QAbstractItemView { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #c27c2a, stop:1 #b06a1f); color: #ffffff; selection-background-color: #f0c57a; selection-color: #1b1208; border: 1px solid #6b4523; }
                """
            )
        except Exception:
            pass

        # Palm sliders matching voxcscript parameters
        self.palm_size_slider, self.palm_size_range = self._create_float_slider(controls_layout, "Size", 0.1, 2.0, 1.0)
        self.palm_trunkextend_slider, self.palm_trunkextend_range = self._create_float_slider(controls_layout, "Trunk Height", 0.0, 340.0, 80.0)
        self.palm_trunksize_slider, self.palm_trunksize_range = self._create_float_slider(controls_layout, "Trunk Width", 0.3, 4.0, 1.0)
        self.palm_trunkiter_slider, self.palm_trunkiter_range = self._create_float_slider(controls_layout, "Trunk Iter", 12, 80, 40)
        self.palm_bend_slider, self.palm_bend_range = self._create_float_slider(controls_layout, "Bend", 0.0, 1.0, 1.0)
        self.palm_leafcount_slider, self.palm_leafcount_range = self._create_float_slider(controls_layout, "Frond Count", 4, 72, 33)
        # Add value display for leaf count
        self.palm_leafcount_value = QLabel(f"{int(self._map_slider_to_range(self.palm_leafcount_slider.value(), 0, 1000, 4, 72))}")
        self.palm_leafcount_value.setStyleSheet("color: #e9d8c1; font-weight: 500;")
        controls_layout.addWidget(self.palm_leafcount_value)
        self.palm_leafcount_slider.valueChanged.connect(lambda: self.palm_leafcount_value.setText(f"{int(self._map_slider_to_range(self.palm_leafcount_slider.value(), 0, 1000, 4, 72))}"))
        self.palm_leaflength_slider, self.palm_leaflength_range = self._create_float_slider(controls_layout, "Frond Length", 0.1, 3.0, 0.35)
        self.palm_leafvar_slider, self.palm_leafvar_range = self._create_float_slider(controls_layout, "Frond Var", 0.0, 1.0, 0.5)
        self.palm_frondrandom_slider, self.palm_frondrandom_range = self._create_float_slider(controls_layout, "Frond Random", 0.0, 1.0, 1.0)
        self.palm_gravity_slider, self.palm_gravity_range = self._create_float_slider(controls_layout, "Gravity", 0.0, 1.0, 0.3)
        self.palm_leafwidth_slider, self.palm_leafwidth_range = self._create_float_slider(controls_layout, "Frond Width", 0.1, 1.0, 0.25)
        # Palm-themed slider style (matches palm button colors)
        palm_slider_style = """
        QSlider { background: transparent; }
        QSlider::groove:horizontal { height: 8px; background: #5a3e21; border-radius: 4px; }
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #c27c2a, stop:1 #b06a1f);
            border: 1px solid #8a5a32;
            width: 18px; margin: -5px 0; border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #d29439, stop:1 #c07a2a);
        }
        """
        # Apply style to all palm sliders
        for slider in [
            self.palm_size_slider, self.palm_trunkextend_slider, self.palm_trunksize_slider,
            self.palm_trunkiter_slider, self.palm_bend_slider, self.palm_leafcount_slider,
            self.palm_leaflength_slider, self.palm_leafvar_slider, self.palm_frondrandom_slider,
            self.palm_gravity_slider, self.palm_leafwidth_slider
        ]:
            slider.setStyleSheet(palm_slider_style)

        # Seed (slider 1..9999)
        self.palm_seed_slider, self.palm_seed_range = self._create_float_slider(controls_layout, "Seed", 1, 9999, 1)
        # apply palm slider style already defined above
        self.palm_seed_slider.setStyleSheet(palm_slider_style)

        self.palm_progress = QProgressBar()
        self.palm_progress.setVisible(False)
        controls_layout.addWidget(self.palm_progress)
        self.palm_preview_progress = QProgressBar()
        self.palm_preview_progress.setVisible(False)
        self.palm_preview_progress.setRange(0,100)
        controls_layout.addWidget(self.palm_preview_progress)

        controls_layout.addStretch(1)
        content_layout.addWidget(controls_panel, 1)

        # Preview column
        preview_layout = QVBoxLayout()
        preview_frame_widget = QWidget()
        preview_frame_widget.setLayout(preview_layout)

        btn_container = QFrame()
        btn_container.setStyleSheet("background: #2b2012; border: 1px solid #4a3220; border-radius: 6px;")
        btn_container_layout = QVBoxLayout(btn_container)
        btn_container_layout.setContentsMargins(8,6,8,6)
        btn_container_layout.setSpacing(6)
        # Palm-themed button style (warm orange/brown gradient)
        _palm_btn_style = (
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #c27c2a, stop:1 #b06a1f);"
            " color: white; padding: 6px 12px; border: none; border-radius: 4px; font-weight: 600; }"
            "QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #d29439, stop:1 #c07a2a); }"
            "QPushButton:disabled { background: #8a5a32; color: #e6d3b8; }"
        )

        # create buttons and apply style
        top_row = QHBoxLayout()
        top_row.addStretch(1)
        self.palm_randomize_btn = QPushButton("Randomize")
        self.palm_randomize_btn.setStyleSheet(_palm_btn_style)
        self.palm_randomize_btn.clicked.connect(self._randomize_palm_sliders)
        top_row.addWidget(self.palm_randomize_btn)

        self.palm_generate_btn = QPushButton("Generate Palm")
        self.palm_generate_btn.setStyleSheet(_palm_btn_style)
        self.palm_generate_btn.clicked.connect(self._on_palm_generate)
        top_row.addWidget(self.palm_generate_btn)
        top_row.addStretch(1)
        btn_container_layout.addLayout(top_row)
        chk_row = QHBoxLayout()
        chk_row.addStretch(1)
        self.palm_open_after = QCheckBox("Open file after generation")
        self.palm_open_after.setChecked(True)
        self.palm_open_after.setStyleSheet(
            """
            QCheckBox { color: #e9d8c1; spacing: 6px; }
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #6b4523; background: #1b1208; }
            QCheckBox::indicator:unchecked { background: #1b1208; }
            QCheckBox::indicator:checked { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #f0c57a, stop:1 #d79a3c); border: 1px solid #b06a1f; }
            """
        )
        chk_row.addWidget(self.palm_open_after)
        chk_row.addStretch(1)
        btn_container_layout.addLayout(chk_row)
        preview_layout.addWidget(btn_container)

        preview_layout.addStretch(1)
        self.palm_preview = QLabel()
        self.palm_preview.setFixedSize(320,320)
        self.palm_preview.setStyleSheet("background: #0f0b05; border: 1px solid #5a3e21;")
        self.palm_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.palm_dim_label = QLabel("")
        self.palm_dim_label.setStyleSheet("color: #e9d8c1; font-weight: 500;")
        self.palm_dim_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.palm_dim_label)
        preview_layout.addWidget(self.palm_preview, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        preview_layout.addStretch(1)
        content_layout.addWidget(preview_frame_widget)

        self.palm_preview.setText("")

        # Connect controls -> preview timer
        controls = [
            self.palm_size_slider, self.palm_trunkextend_slider, self.palm_trunksize_slider, self.palm_trunkiter_slider,
            self.palm_bend_slider, self.palm_leafcount_slider, self.palm_leaflength_slider, self.palm_leafvar_slider,
            self.palm_frondrandom_slider, self.palm_gravity_slider, self.palm_leafwidth_slider, self.palm_seed_slider, self.palm_palette_combo
        ]
        for c in controls:
            if isinstance(c, QSlider):
                c.valueChanged.connect(lambda _: self.palm_preview_timer.start(300))
            elif isinstance(c, QSpinBox):
                c.valueChanged.connect(lambda _: self.palm_preview_timer.start(300))
            elif isinstance(c, QComboBox):
                c.currentIndexChanged.connect(lambda _: self.palm_preview_timer.start(300))

        # start initial preview
        self.palm_preview_timer.start(100)

    def _start_palm_preview(self):
        if not PALM_AVAILABLE:
            return
        def s2r(slider, r):
            v = slider.value()
            amin, amax = r
            return amin + (amax - amin) * (v / 1000.0)
        params = {
            'size': s2r(self.palm_size_slider, self.palm_size_range),
            'trunkextend': s2r(self.palm_trunkextend_slider, self.palm_trunkextend_range),
            'trunksize': s2r(self.palm_trunksize_slider, self.palm_trunksize_range),
            'trunkiter': int(s2r(self.palm_trunkiter_slider, self.palm_trunkiter_range)),
            'bend': s2r(self.palm_bend_slider, self.palm_bend_range),
            'leafcount': int(s2r(self.palm_leafcount_slider, self.palm_leafcount_range)),
            'leaflength': s2r(self.palm_leaflength_slider, self.palm_leaflength_range),
            'leafvar': s2r(self.palm_leafvar_slider, self.palm_leafvar_range),
            'frondrandom': s2r(self.palm_frondrandom_slider, self.palm_frondrandom_range),
            'gravity': s2r(self.palm_gravity_slider, self.palm_gravity_range),
            'leafwidth': s2r(self.palm_leafwidth_slider, self.palm_leafwidth_range),
            'seed': int(s2r(self.palm_seed_slider, self.palm_seed_range))
        }
        palette = self.palm_palette_combo.currentText() if self.palm_palette_combo.count() > 0 else ''
        if self._preview_thread is not None and self._preview_thread.isRunning():
            try:
                self._preview_thread.cancel()
            except Exception:
                pass
        thr = PreviewThread('palm', params, palette, self._export_executor, parent=self)
        self._register_preview_thread(thr)
        self._preview_thread = thr
        thr.progress.connect(self._on_palm_preview_progress)
        thr.finished.connect(self._on_preview_ready)
        thr.error.connect(self._on_preview_error)
        thr.finished.connect(lambda *_args, t=thr: self._unregister_preview_thread(t))
        thr.error.connect(lambda *_args, t=thr: self._unregister_preview_thread(t))
        self.palm_preview_progress.setRange(0,0)
        self.palm_preview_progress.setVisible(True)
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

    def _on_birch_preview_progress(self, p):
        try:
            self.birch_preview_progress.setValue(int(p * 100))
        except Exception:
            pass

    def _on_palm_preview_progress(self, p):
        try:
            self.palm_preview_progress.setValue(int(p * 100))
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
            elif mode == 'pine':
                self.pine_preview.setPixmap(pix.scaled(self.pine_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                self.pine_preview_progress.setVisible(False)
            elif mode == 'birch':
                self.birch_preview.setPixmap(pix.scaled(self.birch_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                self.birch_preview_progress.setVisible(False)
            elif mode == 'palm':
                self.palm_preview.setPixmap(pix.scaled(self.palm_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                self.palm_preview_progress.setVisible(False)
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
                try:
                    self.birch_preview_progress.setVisible(False)
                except Exception:
                    pass
                try:
                    self.palm_preview_progress.setVisible(False)
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
                "Version 1.5.2\n\n"
                "Made by NGNT Creations\n"
                "Preview and export voxel trees using the treegen-pinegen library.\n"
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
            if isinstance(child, QLabel) and (child is getattr(self, 'tree_preview', None) or child is getattr(self, 'pine_preview', None) or child is getattr(self, 'birch_preview', None) or child is getattr(self, 'palm_preview', None)):
                continue
            try:
                child.setDisabled(disable)
            except Exception:
                pass
    def _format_output_path(self, filename, max_len=48):
        """Return a shortened, GUI-friendly path for display.
        Prefer a path starting at the 'output' directory when present, otherwise use a relative
        path. If still too long, truncate the middle keeping start and end.
        """
        try:
            if not filename:
                return ''
            # prefer the portion starting with 'output' to avoid long absolute prefixes
            norm = os.path.normpath(filename)
            parts = norm.split(os.sep)
            if 'output' in parts:
                idx = parts.index('output')
                short = os.path.join(*parts[idx:])
            else:
                try:
                    short = os.path.relpath(norm, start=os.getcwd())
                except Exception:
                    short = os.path.basename(norm)
            if len(short) <= max_len:
                return short
            # truncate middle
            head = short[: max_len//2 - 2]
            tail = short[-(max_len//2 - 1):]
            return head + '...' + tail
        except Exception:
            try:
                return os.path.basename(filename)
            except Exception:
                return filename

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
            'seed': int(self._map_slider_to_range(self.tree_seed_slider.value(), 0, 1000, 1, 9999)),
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
            'seed': int(self._map_slider_to_range(self.pine_seed_slider.value(), 0, 1000, 1, 9999))
        }
        palette = self.pine_palette_combo.currentText() if self.pine_palette_combo.count() > 0 else ''
        self.pine_progress.setVisible(True)
        self.pine_progress.setRange(0, 0)
        self._disable_tab_controls(self.pine_tab, True)
        future = self._export_executor.submit(export_pine, params, palette)
        future.add_done_callback(lambda f, m='pine': self._handle_export_future(f, m))

    def _on_birch_generate(self):
        if not BIRCH_AVAILABLE:
            QMessageBox.critical(self, 'Error', 'Birch generation not available')
            return
        size = self._map_slider_to_range(self.birch_size_slider.value(), 0, 1000, 0.1, 3.0)
        params = {
            'size': size,
            'twisted': self._map_slider_to_range(self.birch_twist_slider.value(), 0, 1000, 0.0, 1.0),
            'trunksize': self._map_slider_to_range(self.birch_trunk_slider.value(), 0, 1000, 0.1, 1.2),
            'trunkheight': 1.0,
            'branchdensity': 1.0,
            'branchlength': 1.0,
            'branchdir': -0.5,
            'leaves': self._map_slider_to_range(self.birch_leaves_slider.value(), 0.0, 1000, 0.0, 3.0),
            'leaf_radius': 2.0,
            'leaf_stretch': 1.5,
            'leaf_bias': -0.3,
            'seed': int(self._map_slider_to_range(self.birch_seed_slider.value(), 0, 1000, 1, 9999)),
            'iterations': self.birch_iterations_spin.value(),
            'spread': self._map_slider_to_range(self.birch_spread_slider.value(), 0, 1000, 0.0, 1.0),
            'gravity': self._map_slider_to_range(self.birch_gravity_slider.value(), 0, 1000, -1.0, 1.0),
            'wide': self._map_slider_to_range(self.birch_wide_slider.value(), 0, 1000, 0.0, 1.0)
        }
        palette = self.birch_palette_combo.currentText() if self.birch_palette_combo.count() > 0 else ''
        self.birch_progress.setVisible(True)
        self.birch_progress.setRange(0, 0)
        self._disable_tab_controls(self.birch_tab, True)
        future = self._export_executor.submit(export_birch, params, palette)
        future.add_done_callback(lambda f, m='birch': self._handle_export_future(f, m))
    def _on_palm_generate(self):
        if not PALM_AVAILABLE:
            QMessageBox.critical(self, 'Error', 'Palm core not available')
            return
        params = {
            'size': self._map_slider_to_range(self.palm_size_slider.value(), 0, 1000, 0.1, 3.0),
            'trunkextend': self._map_slider_to_range(self.palm_trunkextend_slider.value(), 0, 1000, 0.0, 340.0),
            'trunksize': self._map_slider_to_range(self.palm_trunksize_slider.value(), 0, 1000, 0.3, 4.0),
            'trunkiter': int(self._map_slider_to_range(self.palm_trunkiter_slider.value(), 0, 1000, 12, 80)),
            'bend': self._map_slider_to_range(self.palm_bend_slider.value(), 0, 1000, 0.0, 1.0),
            'leafcount': int(self._map_slider_to_range(self.palm_leafcount_slider.value(), 0, 1000, 4, 72)),
            'leaflength': self._map_slider_to_range(self.palm_leaflength_slider.value(), 0, 1000, 0.1, 3.0),
            'leafvar': self._map_slider_to_range(self.palm_leafvar_slider.value(), 0, 1000, 0.0, 1.0),
            'frondrandom': self._map_slider_to_range(self.palm_frondrandom_slider.value(), 0, 1000, 0.0, 1.0),
            'gravity': self._map_slider_to_range(self.palm_gravity_slider.value(), 0, 1000, 0.0, 1.0),
            'leafwidth': self._map_slider_to_range(self.palm_leafwidth_slider.value(), 0, 1000, 0.1, 1.0),
            'seed': int(self._map_slider_to_range(self.palm_seed_slider.value(), 0, 1000, 1, 9999)),
        }
        palette = self.palm_palette_combo.currentText() if self.palm_palette_combo.count() > 0 else ''
        self.palm_progress.setVisible(True)
        self.palm_progress.setRange(0,0)
        self._disable_tab_controls(self.palm_tab, True)
        future = self._export_executor.submit(export_palm, params, palette)
        future.add_done_callback(lambda f, m='palm': self._handle_export_future(f, m))
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
        elif mode == 'pine':
            self.pine_progress.setVisible(False)
            self._disable_tab_controls(self.pine_tab, False)
            self.pine_generate_btn.setEnabled(True)
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
        elif mode == 'birch':
            self.birch_progress.setVisible(False)
            self._disable_tab_controls(self.birch_tab, False)
            self.birch_generate_btn.setEnabled(True)
            try:
                display = self._format_output_path(filename)
                self.birch_dim_label.setText(f'Generated: {display}')
            except Exception:
                pass
            if self.birch_open_after.isChecked():
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
        elif mode == 'palm':
            self.palm_progress.setVisible(False)
            self._disable_tab_controls(self.palm_tab, False)
            self.palm_generate_btn.setEnabled(True)
            try:
                display = self._format_output_path(filename)
                self.palm_dim_label.setText(f'Generated: {display}')
            except Exception:
                pass
            if self.palm_open_after.isChecked():
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
        elif mode == 'pine':
            self.pine_progress.setVisible(False)
            self._disable_tab_controls(self.pine_tab, False)
            self.pine_generate_btn.setEnabled(True)
        elif mode == 'birch':
            self.birch_progress.setVisible(False)
            self._disable_tab_controls(self.birch_tab, False)
            self.birch_generate_btn.setEnabled(True)
        elif mode == 'palm':
            self.palm_progress.setVisible(False)
            self._disable_tab_controls(self.palm_tab, False)
            self.palm_generate_btn.setEnabled(True)
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
        set_slider_from_range(self.tree_wide_slider, self.tree_wide_range)
        # iterations spinbox
        self.tree_iterations_spin.setValue(random.randint(self.tree_iterations_spin.minimum(), self.tree_iterations_spin.maximum()))
        # seed (slider)
        set_slider_from_range(self.tree_seed_slider, self.tree_seed_range)
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
        set_slider_from_range(self.pine_seed_slider, self.pine_seed_range)
        # palette is not randomized here; set manually in UI
        # trigger preview
        self.pine_preview_timer.start(50)

    def _randomize_birch_sliders(self):
        def set_slider_from_range(slider, r):
            amin, amax = r
            v = random.uniform(amin, amax)
            t = int(round((v - amin) / (amax - amin) * 1000)) if amax != amin else 0
            slider.setValue(max(0, min(1000, t)))

        set_slider_from_range(self.birch_size_slider, self.birch_size_range)
        set_slider_from_range(self.birch_trunk_slider, self.birch_trunk_range)
        set_slider_from_range(self.birch_spread_slider, self.birch_spread_range)
        set_slider_from_range(self.birch_twist_slider, self.birch_twist_range)
        set_slider_from_range(self.birch_leaves_slider, self.birch_leaves_range)
        set_slider_from_range(self.birch_gravity_slider, self.birch_gravity_range)
        set_slider_from_range(self.birch_wide_slider, self.birch_wide_range)
        self.birch_iterations_spin.setValue(random.randint(self.birch_iterations_spin.minimum(), self.birch_iterations_spin.maximum()))
        set_slider_from_range(self.birch_seed_slider, self.birch_seed_range)
        self.birch_preview_timer.start(50)
    
    def _randomize_palm_sliders(self):
        def set_slider_from_range(slider, r):
            amin, amax = r
            v = random.uniform(amin, amax)
            t = int(round((v - amin) / (amax - amin) * 1000)) if amax != amin else 0
            slider.setValue(max(0, min(1000, t)))
        set_slider_from_range(self.palm_size_slider, self.palm_size_range)
        set_slider_from_range(self.palm_trunkextend_slider, self.palm_trunkextend_range)
        set_slider_from_range(self.palm_trunksize_slider, self.palm_trunksize_range)
        set_slider_from_range(self.palm_trunkiter_slider, self.palm_trunkiter_range)
        set_slider_from_range(self.palm_bend_slider, self.palm_bend_range)
        set_slider_from_range(self.palm_leafcount_slider, self.palm_leafcount_range)
        set_slider_from_range(self.palm_leaflength_slider, self.palm_leaflength_range)
        set_slider_from_range(self.palm_leafvar_slider, self.palm_leafvar_range)
        set_slider_from_range(self.palm_frondrandom_slider, self.palm_frondrandom_range)
        set_slider_from_range(self.palm_gravity_slider, self.palm_gravity_range)
        set_slider_from_range(self.palm_leafwidth_slider, self.palm_leafwidth_range)
        set_slider_from_range(self.palm_seed_slider, self.palm_seed_range)
        self.palm_preview_timer.start(50)

    def _on_tab_changed(self, index):
        # Stop all previews when switching tabs
        try:
            self.tree_preview_timer.stop()
            self.pine_preview_timer.stop()
            self.birch_preview_timer.stop()
            self.palm_preview_timer.stop()
        except Exception:
            pass

        # Restart the preview timer for the newly selected tab, if applicable
        try:
            if index == 0:  # Treegen tab
                self.tree_preview_timer.start(100)
            elif index == 1:  # Pinegen tab
                self.pine_preview_timer.start(100)
            elif index == 2:  # Birchgen tab
                self.birch_preview_timer.start(100)
            elif index == 3:  # Palmgen tab
                self.palm_preview_timer.start(100)
        except Exception:
            pass

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