# 🌲 treegen-pinegen

![treegen-pinegen logo](https://cdn.nostrcheck.me/46025249f65d47dddb0f17d93eb8b0a32d97fe3189c6684bbd33136a0a7e0424/ee9d2b2b47e1bb1656f784e78fcf0db54449afc8b3ac8969c54ca9c3cbdf4afa.webp)

Procedural Voxel Tree + Pine Tree + Birch Tree Generator for MagicaVoxel

Generate customizable .vox trees using palettes, sliders, and pure Python.

Built with Python, NumPy, and Pillow. The project uses a PyQt6 GUI (`treegen_qt.py`) which calls the shared core in worker modules.

## ✨ Features

- Three generators
  - Treegen — oak-style branching tree generator
  - Pinegen — pine/conifer generator with cone-shaped leaf clusters
  - Birchgen — birch tree generator with slender trunks and spreading crowns
- GUI
  - PyQt6 UI: `treegen_qt.py` (menu, About dialog, inline status, tabbed interface)
- Real-time preview with multiprocessing (heavy generation in subprocesses for speed)
- Export to MagicaVoxel `.vox` via `VoxExporter`
- Custom palettes: use 256-color PNG palettes placed in `palettes/tree/`, `palettes/pine/`, `palettes/birch/`
- Randomize presets and deterministic generation via seed
- Organized output folders: `output/tree/`, `output/pine/`, `output/birch/`
- "Open file after generation" option (platform-aware)
- Non-modal export status displayed above each preview in the PyQt UI
- Unique filenames using timestamps to avoid overwrites

## Controls (implemented in the PyQt UI)

Treegen sliders/controls and ranges:
- `Size`: 0.1 — 3.0
- `Trunk Size`: 0.1 — 1.2
- `Spread`: 0.0 — 1.0
- `Twist` (twisted): 0.0 — 1.0
- `Leafiness`: 0.0 — 3.0
- `Gravity`: -1.0 — 1.0
- `Wide`: 0.0 — 1.0
- `Iterations`: 5 — 15 (integer)
- `Seed`: 1 — 9999

Pinegen sliders/controls and ranges:
- `Size`: 0.1 — 3.0
- `Twist` (twisted): 0.0 — 4.0
- `Trunk Size`: 1.0 — 3.0
- `Trunk Height`: 0.0 — 5.0
- `Branch Density`: 0.0 — 3.0
- `Branch Length`: 0.0 — 3.0
- `Branch Direction`: -5.0 — 5.0
- `Leafiness`: 0.0 — 2.0
- `Leaf Radius`: 1.0 — 4.0
- `Leaf Stretch`: 0.5 — 3.0
- `Leaf Bias`: -1.0 — 1.0
- `Seed`: 1 — 9999

Birchgen sliders/controls and ranges:
- `Size`: 0.1 — 3.0
- `Trunk Size`: 0.05 — 0.8
- `Spread`: 0.1 — 0.8
- `Twist` (twisted): 0.0 — 1.0
- `Leafiness`: 0.0 — 3.0
- `Gravity`: -1.0 — 1.0
- `Wide`: 0.05 — 0.2
- `Iterations`: 5 — 14 (integer)
- `Seed`: 1 — 9999

Other UI features (PyQt6 port)
- Menu bar: File → Close, Help → About (About shows app/version/credits)
- Inline status: export results appear above the preview (`tree_dim_label`/`pine_dim_label`/`birch_dim_label`)
- Preview uses worker modules and runs in a worker QThread with progress signals and cooperative cancellation
- Export runs in a separate process (ProcessPoolExecutor) and writes `.vox` files using `VoxExporter`
- Tabbed interface for easy switching between generators
- Previews update on startup, tab changes, and palette selections

## Code layout
- `treegen_qt.py` — PyQt6 GUI (primary frontend)
- `treegen_worker.py` — Treegen generation logic and VoxExporter
- `pinegen_worker.py` — Pinegen generation logic and VoxExporter
- `birch_worker.py` — Birchgen generation logic and VoxExporter
- `palette_worker.py` — Palette management utilities
- `palettes/` — palette PNGs for tree, pine, and birch
- `output/tree/`, `output/pine/`, `output/birch/` — generated .vox files

## Requirements

Minimum Python packages:
- Python 3.8+
- numpy
- pillow
- PyQt6

Install with pip:

```bash
pip install numpy pillow PyQt6
```

## Run

PyQt6 UI (primary):

```bash
python treegen_qt.py
```

The PyQt UI calls into the worker modules. Use the preview to iterate before exporting `.vox` files.

## Palettes

Each palette is expected to be a 256-entry PNG (one-row palette). Place palettes in:
- `palettes/tree/`
- `palettes/pine/`
- `palettes/birch/`

Palette mapping for leaves/trunk colors is defined in the worker modules (see `TREE_PALETTE_MAP`, `PINE_PALETTE_MAP`, etc.).

## Export & Output

- Exports produce `.vox` files compatible with MagicaVoxel and are saved to `output/tree/`, `output/pine/`, or `output/birch/`.
- Filenames use timestamps (e.g., `treegen_20231005_143022.vox`) to ensure uniqueness without external counter files.

## Credits

Made by NGNT Creations

## License

MIT — Free to use and modify.