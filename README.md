# 🌲 treegen-pinegen

![treegen-pinegen logo](https://cdn.nostrcheck.me/46025249f65d47dddb0f17d93eb8b0a32d97fe3189c6684bbd33136a0a7e0424/fabf53203e2e3dda398dc62bfd730295036117a77891cf8f2908318df20aff08.webp)
![treegen-pinegen logo](https://cdn.nostrcheck.me/46025249f65d47dddb0f17d93eb8b0a32d97fe3189c6684bbd33136a0a7e0424/a91d292209d98d746086ca5f509f33e92c97f6f6f74db41e28c4d85150a91497.webp)
![treegen-pinegen logo](https://cdn.nostrcheck.me/46025249f65d47dddb0f17d93eb8b0a32d97fe3189c6684bbd33136a0a7e0424/52b43ad80f51100004c6dfd41ef822382ea07403b27efbd0f165a2db4156875c.webp)
![treegen-pinegen logo](https://cdn.nostrcheck.me/46025249f65d47dddb0f17d93eb8b0a32d97fe3189c6684bbd33136a0a7e0424/a05c1f2a619809ea2845b35e8f1f42c27a73e936a9c45b06ddf82a376a99b503.webp)
![treegen-pinegen logo](https://cdn.nostrcheck.me/46025249f65d47dddb0f17d93eb8b0a32d97fe3189c6684bbd33136a0a7e0424/a4a90956ae5108a1c68e5323a02bcccce743c6bef2e5b8030b6f9c5e232073ca.webp)

Procedural Voxel Tree Generator for MagicaVoxel

Generate customizable .vox trees using palettes, sliders, and pure Python.

Built with Python, NumPy, and Pillow. The project uses a PyQt6 GUI (`treegen_qt.py`) which calls the shared core in worker modules.

## ✨ Features

- Five generators
  - Treegen — oak-style branching tree generator
  - Pinegen — pine/conifer generator with cone-shaped leaf clusters
  - Birchgen — birch tree generator with slender trunks and spreading crowns
  - Palmgen — tropical palm generator
  - Kapokgen — broadleaf tropical tree generator (new)
- GUI
  - PyQt6 UI: `treegen_qt.py` (menu, About dialog, inline status, tabbed interface)
- Real-time preview with multiprocessing (heavy generation in subprocesses for speed)
- Export to MagicaVoxel `.vox` via `VoxExporter`
- Custom palettes: use 256-color PNG palettes placed in `palettes/tree/`, `palettes/pine/`, `palettes/birch/`, `palettes/palm/`, `palettes/kapok/` (internal palm and kapok palettes added)
- Randomize presets and deterministic generation via seed
- Organized output folders: `output/tree/`, `output/pine/`, `output/birch/`, `output/palm/`, `output/kapok/`
- "Open file after generation" option (platform-aware)
- Non-modal export status displayed above each preview in the PyQt UI
- Unique filenames using timestamps to avoid overwrites

## Recent additions
- Kapokgen — broadleaf tropical tree generator with roots, canopy, and leaves
  - Kapok palette combo popup uses themed styling consistent with other tabs.
  - Kapok seed control uses the themed slider wired to preview, export, and randomize.
- New internal kapok palettes (programmatically registered)
  - `Kapok Tropical` (existing)
  - `Kapok Lush` — denser, deeper greens + bright upper canopy highlights
  - `Kapok Dusk` — warm highlights, yellow/orange top canopy and darker trunk for sunset scenes
  - `Kapok Rainforest` — deep emerald greens with golden trunk flecks for a dense jungle
  - `Kapok Exotic` — electric pinks and purples for a surreal, bioluminescent paradise
  - `Kapok Neon` — glowing blues and cyans like a cyber-tropical forest
  - `Kapok Mystic` — enchanted purples and golds for a magical grove
  - `Kapok Aurora` — shimmering greens and blues like northern lights over a tropical tree
  - `Kapok Inferno` — blazing reds and oranges for a volcanic tropical inferno
  - `Kapok Crystal` — icy blues and whites for a frozen crystalline wonder
  These are provided via `palette_worker.py` so no external PNG files are required to use them.
- Palmgen UI polish
  - Palm palette combo popup now uses the themed styling consistent with other tabs (Tree/Pine/Birch/Kapok).
  - Palm seed control was changed from a spinbox to the themed slider used across the app; the slider value is wired to preview, export and randomize.
- New internal palm palettes (programmatically registered)
  - `Palm Default` (existing)
  - `Palm Lush` — deep saturated tropical greens with darker trunk
  - `Palm Dusk` — cool bluish fronds for low-light scenes
  - `Palm Sandbar` — pale yellow-greens and sandy trunk for beachy palms
  - `Palm Sunset` — warm orange-tinged fronds with rich trunk tones
  These are provided via `palette_worker.py` so no external PNG files are required to use them.
- Frond colour mapping
  - Leaf voxels are now mapped to palette indices based on their fractional position along each frond (t01) so frond mid-lengths read darker and tips read lighter. This produces a natural dark band down the centre of fronds and a smooth outward-lightening gradient.

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

Palmgen sliders/controls and ranges:
- `Size`: 0.1 — 3.0
- `Trunk Height`: 0.0 — 340.0
- `Trunk Width`: 0.3 — 4.0
- `Trunk Iter`: 12 — 80 (integer)
- `Bend`: 0.0 — 1.0
- `Frond Count`: 4 — 72
- `Frond Length`: 0.1 — 3.0
- `Frond Var`: 0.0 — 1.0
- `Frond Random`: 0.0 — 1.0
- `Gravity`: 0.0 — 1.0
- `Frond Width`: 0.1 — 1.0
- `Seed`: 1 — 9999

Kapokgen sliders/controls and ranges (new):
- `Size`: 0.1 — 3.0
- `Trunk Extend`: 0.0 — 340.0
- `Trunk Size`: 0.3 — 4.0
- `Trunk Iter`: 8 — 40 (integer)
- `Bend`: 0.0 — 1.0
- `Root Twist`: 0.0 — 1.0
- `Root Profile`: 0.0 — 1.0
- `Root Spread`: 0.0 — 1.0
- `Root Count`: 0 — 8 (integer)
- `Root Variance`: 0.0 — 1.0
- `Canopy Iter`: 5 — 15 (integer)
- `Wide`: 0.0 — 1.0
- `Spread`: 0.0 — 1.0
- `Canopy Twist`: 0.0 — 1.0
- `Leaves`: 0.0 — 3.0
- `Gravity`: -1.0 — 1.0
- `Canopy Thick`: 0.0 — 1.0
- `Canopy Profile`: 0.0 — 1.0
- `Canopy Flat`: 0.0 — 1.0
- `Canopy Start`: 0.0 — 1.0
- `Seed`: 1 — 9999

Other UI features (PyQt6 port)
- Menu bar: File → Close, Help → About (About shows app/version/credits)
- Inline status: export results appear above the preview (`tree_dim_label`/`pine_dim_label`/`birch_dim_label`/`palm_dim_label`/`kapok_dim_label`)
- Preview uses worker modules and runs in a worker QThread with progress signals and cooperative cancellation
- Export runs in a separate process (ProcessPoolExecutor) and writes `.vox` files using `VoxExporter`
- Tabbed interface for easy switching between generators
- Previews update on startup, tab changes, and palette selections

## Code layout
- `treegen_qt.py` — PyQt6 GUI (primary frontend)
- `treegen_worker.py` — Treegen generation logic and VoxExporter
- `pinegen_worker.py` — Pinegen generation logic and VoxExporter
- `birch_worker.py` — Birchgen generation logic and VoxExporter
- `palm_worker.py` — Palm generator and export logic
- `kapok_worker.py` — Kapok generator and export logic (new)
- `palette_worker.py` — Palette management utilities (internal palm and kapok palettes added)
- `palettes/` — palette PNGs for tree, pine, birch, palm, and kapok (optional when internal palettes are used)
- `output/tree/`, `output/pine/`, `output/birch/`, `output/palm/`, `output/kapok/` — generated .vox files

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
- `palettes/palm/`
- `palettes/kapok/`

Palm and kapok palettes are also provided programmatically inside `palette_worker.py` and do not require external PNG files. Use the Palm and Kapok-themed palettes via the respective tab combos.

## Export & Output

- Exports produce `.vox` files compatible with MagicaVoxel and are saved to `output/tree/`, `output/pine/`, `output/birch/`, `output/palm/`, or `output/kapok/`.
- Filenames use timestamps (e.g., `kapok_20231005_143022.vox`) to ensure uniqueness without external counter files.

## Credits

Made by NGNT Creations

## License

MIT — Free to use and modify.