import os
import random
from datetime import datetime
import math
import struct

from PIL import Image
from PIL import Image as PILImage
try:
    from palette_worker import get_internal_palette, _palette_manager
except Exception:
    # palette_worker may not be present in some environments; fall back to file-based palettes
    get_internal_palette = None
    _palette_manager = None

try:
    import numpy as np
except Exception:
    raise SystemExit('numpy is required for treegen_worker')

# Basic constants matching the main project
GRID = 256
PREVIEW_GRID = 64

# resource helper (PyInstaller compatibility)
import sys as _sys

def resource_path(filename):
    if hasattr(_sys, '_MEIPASS'):
        return os.path.join(_sys._MEIPASS, filename)
    return filename

def clamp(v, mi, ma):
    return max(mi, min(ma, v))

class VoxExporter:
    def __init__(self, params, palette_map=None, palette_subdir='tree', output_subdir='tree'):
        """
        Simplified VoxExporter: no counter file, time-based filenames only.
        Backwards-compatible with callers that pass (params, palette_map, palette_subdir, output_subdir).
        """
        self.params = params
        self.palette_map = palette_map or {'default': {'leaves': [9, 17], 'trunk': [57, 65]}}
        self.palette_subdir = palette_subdir
        self.output_subdir = output_subdir

    def load_palette(self, palette_name):
        """
        Prefer the internal palette registry if available. Do not attempt to load palette files from disk.
        Returns (palette_list, leaf_indices, trunk_indices).
        """
        key = os.path.basename(palette_name) if palette_name else 'default'
        try:
            if get_internal_palette and _palette_manager and key in _palette_manager.list_palettes():
                palette, mapping = get_internal_palette(key)
                return palette, mapping.get('leaves', [9, 17]), mapping.get('trunk', [57, 65])
        except Exception:
            # fallthrough to default palette
            pass

        # Fallback: simple grayscale palette and mapping from provided palette_map
        palette = [(i, i, i, 255) for i in range(256)]
        config = self.palette_map.get(key, next(iter(self.palette_map.values()))) if self.palette_map else {'leaves': [9,17], 'trunk': [57,65]}
        return palette, config.get('leaves', [9, 17]), config.get('trunk', [57, 65])

    def export(self, voxels, palette, leaf_indices, trunk_indices, prefix='treegen', preview=False):
        if preview:
            return voxels, palette
        # Find all non-empty voxel coordinates and compute tight bounding box
        coords = np.argwhere(voxels > 0)
        voxel_data = bytearray()
        if coords.size == 0:
            min_xyz = np.array([0, 0, 0], dtype=int)
            dims = np.array([1, 1, 1], dtype=int)
            count = 0
        else:
            min_xyz = coords.min(axis=0).astype(int)
            max_xyz = coords.max(axis=0).astype(int)
            dims = (max_xyz - min_xyz + 1).astype(int)
            # Build a deterministic sorted list of voxels inside the bounding box
            rel_coords = []
            for x, y, z in coords:
                c = int(voxels[x, y, z])
                x0 = int(x - min_xyz[0])
                y0 = int(y - min_xyz[1])
                z0 = int(z - min_xyz[2])
                rel_coords.append((x0, y0, z0, c))
            # sort by z, then y, then x to be deterministic
            rel_coords.sort(key=lambda t: (t[2], t[1], t[0]))
            for x0, y0, z0, c in rel_coords:
                voxel_data += struct.pack('<4B', x0, y0, z0, c)
            count = len(rel_coords)

        # Clear any voxels outside the bounding box to ensure no invisible voxels
        # Optimize clearing voxels outside the bounding box using NumPy slicing
        coords = np.argwhere(voxels > 0)  # Ensure coords is defined
        if coords.size > 0:
            min_xyz = coords.min(axis=0).astype(int)
            max_xyz = coords.max(axis=0).astype(int)
            mask = np.ones_like(voxels, dtype=bool)
            mask[min_xyz[0]:max_xyz[0]+1, min_xyz[1]:max_xyz[1]+1, min_xyz[2]:max_xyz[2]+1] = False
            voxels[mask] = 0

        # --- Palette index fix for MagicaVoxel/Teardown ---
        # Shift palette left by 1 so palette[8] is index 9 in MagicaVoxel
        if len(palette) >= 256:
            palette = palette[1:256] + [(0, 0, 0, 0)]
        else:
            palette = palette[1:] + [(0, 0, 0, 0)]
        # Only write the first 256 colors (indices 1-256 in .vox)
        palette = palette[:256]

        size_chunk = b'SIZE' + struct.pack('<ii', 12, 0)
        size_chunk += struct.pack('<iii', int(dims[0]), int(dims[1]), int(dims[2]))
        xyzi_payload = struct.pack('<i', count) + voxel_data
        xyzi_chunk = b'XYZI' + struct.pack('<ii', len(xyzi_payload), 0) + xyzi_payload
        rgba_payload = b''.join(struct.pack('<4B', *c) for c in palette)
        rgba_chunk = b'RGBA' + struct.pack('<ii', len(rgba_payload), 0) + rgba_payload
        main_content = size_chunk + xyzi_chunk + rgba_chunk
        main_chunk = b'MAIN' + struct.pack('<ii', 0, len(main_content)) + main_content
        vox_file = b'VOX ' + struct.pack('<i', 150) + main_chunk

        # Use timestamp for unique filename (counter file removed)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join('output', self.output_subdir)
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f'{prefix}_{timestamp}.vox')
        with open(filename, 'wb') as f:
            f.write(vox_file)
        return filename

# Adding the missing palette mappings for TREE_PALETTE_MAP
TREE_PALETTE_MAP = {
    "tree_default.png": {"leaves": [9, 17], "trunk": [57, 65]},
    "tree_basic.png":   {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))},
    "autumn.png":       {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))},
    "birch.png":        {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))},
    "blossom.png":      {"leaves": list(range(9, 25)), "trunk": list(range(57, 65))},
    "dead.png":         {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))},
    "oak1.png":         {"leaves": list(range(9, 17)), "trunk": list(range(65, 73))},
    "oak2.png":         {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))},
    "tree_sapling.png": {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))}
}

# Copying relevant generation logic for treegen tab from treegen_core.py
# - Added `generate_treegen_tree` function for tree generation logic.
# - Added `generate_treegen_preview` function for preview generation.
# - Added `export_tree` function for exporting `.vox` files.
# Copied functions from treegen_core.py

# CancelledError for cooperative cancellation
class CancelledError(Exception):
    pass

def generate_treegen_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=None, cancel_check=None):
    # Use a local RNG seeded from params so we don't mutate global `random` state
    seed = int(params.get('seed', 1))
    rng = random.Random(seed)

    exporter = VoxExporter(params, TREE_PALETTE_MAP, 'tree', 'tree')
    palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name) if palette_name else ([(i,i,i,255) for i in range(256)], [9,17], [57,65])

    voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    gLeaves = []

    iters = max(int(params.get('iterations', 1)), 1)
    size = 150 * params.get('size', 1.0) / iters
    gTrunkSize = params.get('trunksize', 1.0) * params.get('size', 1.0) * 6
    wide = min(params.get('wide', 0.5), 0.95)
    gBranchLength0 = size * (1 - wide)
    gBranchLength1 = size * wide

    def normalize(x, y, z):
        l = math.sqrt(x*x + y*y + z*z)
        return (x/l, y/l, z/l) if l > 0 else (0,0,1)

    trunk_voxels = set()

    def draw_line(x0, y0, z0, x1, y1, z1, r0, r1):
        steps = int(math.dist([x0, y0, z0], [x1, y1, z1]) * 2)
        if steps == 0:
            steps = 1
        for i in range(steps + 1):
            if cancel_check and cancel_check():
                raise CancelledError()
            t = i / steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            z = z0 + t * (z1 - z0)
            r = r0 + t * (r1 - r0)
            for dx in range(-math.ceil(r), math.ceil(r)+1):
                for dy in range(-math.ceil(r), math.ceil(r)+1):
                    for dz in range(-math.ceil(r), math.ceil(r)+1):
                        if dx*dx + dy*dy + dz*dz <= r*r:
                            vx = int(x+dx)
                            vy = int(y+dy)
                            vz = int(z+dz)
                            if 0 <= vx < grid_size and 0 <= vy < grid_size and 0 <= vz < grid_size:
                                trunk_voxels.add((vx, vy, vz))

    def get_branch_length(i):
        t = math.sqrt((i - 1) / iters)
        return gBranchLength0 + t * (gBranchLength1 - gBranchLength0)

    def get_branch_size(i):
        t = math.sqrt((i - 1) / iters)
        return (1 - t) * gTrunkSize

    def get_branch_angle(i):
        t = math.sqrt((i - 1) / iters)
        return 2.0 * params.get('spread', 0.5) * t

    def get_branch_prob(i):
        return math.sqrt((i - 1) / iters)

    progress_total = iters * 10
    progress_done = 0

    def branches(x, y, z, dx, dy, dz, i):
        nonlocal progress_done
        if cancel_check and cancel_check():
            raise CancelledError()
        l = get_branch_length(i)
        s0 = get_branch_size(i)
        s1 = get_branch_size(i+1) if i+1 <= iters else 0
        x1 = x + dx * l
        y1 = y + dy * l
        z1 = z + dz * l
        draw_line(x, y, z, x1, y1, z1, s0, s1)

        if i < iters:
            b = 1
            var = i * 0.2 * params.get('twisted', 0.5)
            if rng.random() < get_branch_prob(i):
                b = 2
                var = get_branch_angle(i)
            for _ in range(b):
                dx2 = dx + rng.uniform(-var, var)
                dy2 = dy + rng.uniform(-var, var)
                dz2 = dz + rng.uniform(-var, var)
                dx2, dy2, dz2 = normalize(dx2, dy2, dz2)
                branches(x1, y1, z1, dx2, dy2, dz2, i + 1)
        else:
            gLeaves.append((x1, y1, z1))
            gLeaves.append(((x + x1)/2, (y + y1)/2, (z + z1)/2))

        progress_done += 1
        if progress_callback:
            progress_callback(min(progress_done / progress_total, 1.0))

    def add_leaves():
        leaf_set = set()
        for pos in gLeaves:
            x1, y1, z1 = map(int, pos)
            for _ in range(int(5 * params.get('leaves', 1.0))):
                x2, y2, z2 = x1, y1, z1
                for _ in range(int(50 * params.get('leaves', 1.0))):
                    leaf_set.add((x2, y2, z2))
                    d = rng.randint(1, 6)
                    if d == 1: x2 -= 1
                    elif d == 2: x2 += 1
                    elif d in (3, 4):
                        z2 += 1 if rng.uniform(-1, 1) < params.get('gravity', 0.0) else -1
                    elif d == 5: y2 -= 1
                    else: y2 += 1
        leaf_list = list(leaf_set)
        rng.shuffle(leaf_list)
        for i, (x, y, z) in enumerate(leaf_list):
            idx = leaf_indices[i % len(leaf_indices)] if leaf_indices else 1
            if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
                if voxels[x, y, z] == 0:
                    voxels[x, y, z] = idx

    branches(grid_size//2, grid_size//2, 0, 0, 0, 1, 1)
    add_leaves()

    trunk_voxels = list(trunk_voxels)
    rng.shuffle(trunk_voxels)
    for i, (x, y, z) in enumerate(trunk_voxels):
        idx = trunk_indices[i % len(trunk_indices)] if trunk_indices else 1
        voxels[x, y, z] = idx

    return voxels, palette

def generate_treegen_preview(params, palette_name, grid_size=PREVIEW_GRID, view='front', progress_callback=None, cancel_check=None):
    try:
        vox, palette = generate_treegen_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=progress_callback, cancel_check=cancel_check)
        img_full = project_voxels_to_image(vox, palette, GRID, view=view)
        return img_full.resize((grid_size * 3, grid_size * 3), Image.NEAREST)
    except CancelledError:
        raise
    except Exception:
        shrink = grid_size / GRID
        params_preview = params.copy()
        params_preview['size'] *= shrink
        params_preview['trunksize'] *= shrink
        vox, palette = generate_treegen_tree(params_preview, palette_name, grid_size=grid_size, preview=True, progress_callback=progress_callback, cancel_check=cancel_check)
        img = project_voxels_to_image(vox, palette, grid_size, view=view)
        return img.resize((grid_size*3, grid_size*3), Image.NEAREST)

def orient_voxels_for_export(voxels, view='front'):
    try:
        if view == 'front':
            return voxels
        elif view == 'top':
            oriented = np.swapaxes(voxels, 0, 2).copy()
            return oriented
        else:
            return voxels
    except Exception:
        return voxels

def export_tree(params, palette_name, prefix='treegen', export_view='front'):
    voxels, palette = generate_treegen_tree(params, palette_name, grid_size=GRID, preview=True)
    # reorient voxels so exported file front matches preview front
    voxels_oriented = orient_voxels_for_export(voxels, view=export_view)
    exporter = VoxExporter(params, TREE_PALETTE_MAP, 'tree', 'tree')
    loaded_palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name) if palette_name else (palette, [9,17], [57,65])
    return exporter.export(voxels_oriented, loaded_palette, leaf_indices, trunk_indices, prefix, preview=False)

# project_voxels_to_image moved here
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