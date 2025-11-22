# 🌲 treegen-pinegen

![treegen-pinegen logo](https://cdn.nostrcheck.me/46025249f65d47dddb0f17d93eb8b0a32d97fe3189c6684bbd33136a0a7e0424/ee9d2b2b47e1bb1656f784e78fcf0db54449afc8b3ac8969c54ca9c3cbdf4afa.webp)

Procedural Voxel Tree + Pine Tree Generator for MagicaVoxel

Generate customizable .vox trees using palettes, sliders, and pure Python.

Built with Python, NumPy, and Pillow. The project now uses a PyQt6 GUI (`treegen_qt.py`) which calls the shared core in `treegen_core.py`.

## ✨ Features

- Two generators
  - Treegen — oak-style branching tree generator
  - Pinegen — pine/conifer generator with cone-shaped leaf clusters
- GUI
  - PyQt6 UI: `treegen_qt.py` (menu, About dialog, inline status)
- Real-time preview (low-res projection) with progress and cancellation
- Export to MagicaVoxel `.vox` via `VoxExporter`
- Custom palettes: use 256-color PNG palettes placed in `palettes/tree/` and `palettes/pine/`
- Randomize presets and deterministic generation via seed
- Organized output folders: `output/tree/` and `output/pine/`
- "Open file after generation" option (platform-aware)
- Non-modal export status displayed above each preview in the PyQt UI

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

Other UI features (PyQt6 port)
- Menu bar: File → Close, Help → About (About shows app/version/credits)
- Inline status: export results appear above the preview (`tree_dim_label`/`pine_dim_label`)
- Preview uses `treegen_core.generate_*_preview` and runs in a worker QThread with progress signals and cooperative cancellation
- Export runs in a separate process (ProcessPoolExecutor) and writes `.vox` files using `VoxExporter` (counter files: `treegen_counter.txt`, `pinegen_counter.txt`)

## Code layout
- `treegen_qt.py` — PyQt6 GUI (primary frontend)
- `treegen_core.py` — shared generation logic, preview builders, and `VoxExporter`
- `palettes/` — palette PNGs for tree and pine
- `output/tree/`, `output/pine/` — generated .vox files

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

The PyQt UI calls into the shared core. Use the preview to iterate before exporting `.vox` files.

## Palettes

Each palette is expected to be a 256-entry PNG (one-row palette). Place palettes in:
- `palettes/tree/`
- `palettes/pine/`

Palette mapping for leaves/trunk colors is defined in the core (see `TREE_PALETTE_MAP` and `PINE_PALETTE_MAP` in `treegen_core.py`).

## Export & Output

- Exports produce `.vox` files compatible with MagicaVoxel and are saved to `output/tree/` or `output/pine/`.
- The exporter maintains a simple counter file (`treegen_counter.txt` / `pinegen_counter.txt`) to avoid overwriting files.

## Credits

Made by NGNT Creations

## License

MIT — Free to use and modify.