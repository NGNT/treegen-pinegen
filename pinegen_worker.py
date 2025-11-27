# (full file content with local-RNG changes)
import os
import random
import math
import struct
from PIL import Image
import numpy as np
from datetime import datetime
try:
    from palette_worker import get_internal_palette, _palette_manager
except Exception:
    get_internal_palette = None
    _palette_manager = None

# Basic constants matching the main project
GRID = 256
PREVIEW_GRID = 64

# resource helper (PyInstaller compatibility)
def resource_path(filename):
    if hasattr(__import__('sys'), '_MEIPASS'):
        return os.path.join(__import__('sys')._MEIPASS, filename)
    return filename

def clamp(v, mi, ma):
    return max(mi, min(ma, v))

# PINE_PALETTE_MAP moved here
PINE_PALETTE_MAP = {
    "pine_default.png": {"leaves": [9, 17], "trunk": [57, 65]},
    "pine_basic.png":   {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))},
    "redpine.png":      {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))},
    "pine_sapling.png": {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))},
    "scotspine.png":    {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))}
}

# CancelledError class moved here
class CancelledError(Exception):
    pass

# VoxExporter class reused
class VoxExporter:
    def __init__(self, params, palette_map=None, palette_subdir='pine', output_subdir='pine', counter_file='pinegen_counter.txt'):
        self.params = params
        def _default_palette_map(key):
            return {'leaves':[9,17],'trunk':[57,65]}
        self.palette_map = palette_map or {'default': _default_palette_map('default')}
        self.palette_subdir = palette_subdir
        self.output_subdir = output_subdir
        self.counter_file = counter_file

    def load_palette(self, palette_name):
        key = os.path.basename(palette_name) if palette_name else 'default'
        try:
            if get_internal_palette and _palette_manager and key in _palette_manager.list_palettes():
                palette, mapping = get_internal_palette(key)
                return palette, mapping.get('leaves', [9, 17]), mapping.get('trunk', [57, 65])
        except Exception:
            pass

        path = resource_path(os.path.join('palettes', self.palette_subdir, key))
        palette = load_palette_png(path) if palette_name else [(i, i, i, 255) for i in range(256)]
        config = self.palette_map.get(key, next(iter(self.palette_map.values())))
        return palette, config.get('leaves', [9, 17]), config.get('trunk', [57, 65])

    def export(self, voxels, palette, leaf_indices, trunk_indices, prefix='pinegen', preview=False):
        if preview:
            return voxels, palette
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
            rel_coords = []
            for x, y, z in coords:
                c = int(voxels[x, y, z])
                x0 = int(x - min_xyz[0])
                y0 = int(y - min_xyz[1])
                z0 = int(z - min_xyz[2])
                rel_coords.append((x0, y0, z0, c))
            rel_coords.sort(key=lambda t: (t[2], t[1], t[0]))
            for x0, y0, z0, c in rel_coords:
                voxel_data += struct.pack('<4B', x0, y0, z0, c)
            count = len(rel_coords)

        # --- Palette index fix for MagicaVoxel/Teardown ---
        # Shift palette left by 1 so palette[8] is index 9 in MagicaVoxel
        if len(palette) >= 256:
            palette = palette[1:256] + [(0, 0, 0, 0)]
        else:
            palette = palette[1:] + [(0, 0, 0, 0)]
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

        # Use timestamp for unique filename instead of counter file
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join('output', self.output_subdir)
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f'{prefix}_{timestamp}.vox')
        with open(filename, 'wb') as f:
            f.write(vox_file)
        return filename

# helper to convert voxel volume to 2D projected image (same as in other workers)
def project_voxels_to_image(voxels, palette, grid_size, view='side'):
    # proj chooses the axis to collapse to 2D for display
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
    from PIL import Image
    return Image.fromarray(img_arr, 'RGBA')


# generate_pinegen_tree moved here
def generate_pinegen_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=None, cancel_check=None):
    # Use local RNG to avoid mutating global random state
    seed = int(params.get('seed', 1))
    rng = random.Random(seed)

    exporter = VoxExporter(params, PINE_PALETTE_MAP, 'pine', 'pine', 'pinegen_counter.txt')
    palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name) if palette_name else ([(i,i,i,255) for i in range(256)], [9,17], [57,65])

    voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    trunk_vox = set()
    leaf_vox = set()
    gLeaves = []

    size = clamp(params.get('size', 1.0), 0.1, 3.0)
    twist_param = clamp(params.get('twisted', 0.5), 0.0, 4.0)
    # Reduce twist effect for larger trees so max-scale trees don't become over-twisted.
    try:
        twist_effect = twist_param / math.sqrt(max(size, 0.0001))
    except Exception:
        twist_effect = twist_param
    trunkheight = params.get('trunkheight', 1.0) * 10
    density = clamp(params.get('branchdensity', 1.0), 0, 3) * 30
    branchlength = clamp(params.get('branchlength', 1.0), 0, 3) * size * 20
    branchdir = clamp(params.get('branchdir', -0.5), -5, 5)
    leaves = clamp(params.get('leaves', 1.0), 0, 2)
    trunk_width = size * params.get('trunksize', 2) * 1.1
    max_iter = math.floor(100 * size / 5)
    fixed_size = 5

    def draw_line(x0, y0, z0, x1, y1, z1, r):
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
            for dx in range(-math.ceil(r), math.ceil(r)+1):
                for dy in range(-math.ceil(r), math.ceil(r)+1):
                    for dz in range(-math.ceil(r), math.ceil(r)+1):
                        if dx*dx + dy*dy + dz*dz <= r*r:
                            xi, yi, zi = int(x+dx), int(y+dy), int(z+dz)
                            if 0 <= xi < grid_size and 0 <= yi < grid_size and 0 <= zi < grid_size:
                                trunk_vox.add((xi, yi, zi))

    def normalize(x, y, z):
        l = math.sqrt(x*x + y*y + z*z)
        return (x/l, y/l, z/l) if l > 0 else (0, 1, 0)

    def get_branch_size(i):
        t = (i - 1) / max_iter
        return (1 - t * t) * trunk_width

    def branch(x, y, z, dx, dy, dz, l):
        steps = math.ceil(l / 3)
        l = l / steps
        for _ in range(steps):
            if cancel_check and cancel_check():
                raise CancelledError()
            x1 = x + dx * l
            y1 = y + dy * l
            z1 = z + dz * l
            dx += rng.uniform(-1/steps, 1/steps)
            dy += rng.uniform(-1/steps, 1/steps) + 0.4 / steps
            dz += rng.uniform(-1/steps, 1/steps)
            dx, dy, dz = normalize(dx, dy, dz)
            draw_line(x, y, z, x1, y1, z1, 0)
            gLeaves.append((x1, y1, z1))
            x, y, z = x1, y1, z1

    def generate_branches(x, y, z, dx, dy, dz, i):
        l = fixed_size
        s0 = get_branch_size(i)
        x1 = x + dx * l
        y1 = y + dy * l
        z1 = z + dz * l
        draw_line(x, y, z, x1, y1, z1, s0)

        if y1 > trunkheight:
            b = (1.0 - i / max_iter) * density + 1
            for _ in range(int(b)):
                a = rng.uniform(0.0, math.tau)
                idx = math.cos(a)
                idy = rng.uniform(0.5, 1.0) * branchdir
                idz = math.sin(a)
                idx, idy, idz = normalize(idx, idy, idz)
                il = (1.0 - i / max_iter) * branchlength * rng.uniform(0.5, 1.5)
                t = rng.uniform(0.0, 1.0)
                x2 = x + (x1 - x) * t
                y2 = y + (y1 - y) * t
                z2 = z + (z1 - z) * t
                branch(x2, y2, z2, idx, idy, idz, il + 3)

        if i < max_iter:
            t = i / max_iter
            var = twist_effect * 0.5 * t * (1 - t)
            dx2 = dx + rng.uniform(-var, var)
            dy2 = dy + rng.uniform(-var, var)
            dz2 = dz + rng.uniform(-var, var)
            dx2, dy2, dz2 = normalize(dx2, dy2, dz2)
            generate_branches(x1, y1, z1, dx2, dy2, dz2, i + 1)
        else:
            gLeaves.append((x1, y1, z1))
            gLeaves.append(((x + x1)/2, (y + y1)/2, (z + z1)/2))

    def generate_leaves():
        radius = int(clamp(params.get('leaf_radius', 2), 1, 4))
        vertical_stretch = clamp(params.get('leaf_stretch', 1.5), 0.1, 5.0)
        direction_bias = clamp(params.get('leaf_bias', -0.3), -1.0, 1.0)
        num_clusters = int(len(gLeaves) * clamp(leaves, 0.1, 2.0))
        sphere_density = max(1, int(4 * leaves))
        sources = rng.sample(gLeaves, min(num_clusters, len(gLeaves))) if gLeaves else []

        for x, y, z in sources:
            cx, cy, cz = int(x), int(y), int(z)
            for _ in range(sphere_density):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        for dz in range(-radius, radius + 1):
                            dist = (dx**2 + dz**2 + (dy * vertical_stretch)**2)
                            if dist <= radius**2:
                                if direction_bias < 0 and dy > 0:
                                    continue
                                if direction_bias > 0 and dy < 0:
                                    continue
                                if rng.random() < 0.3:
                                    continue
                                lx, ly, lz = cx + dx, cy + dy, cz + dz
                                if 0 <= lx < grid_size and 0 <= ly < grid_size and 0 <= lz < grid_size:
                                    if voxels[lx, ly, lz] == 0:
                                        leaf_vox.add((lx, ly, lz))

    generate_branches(grid_size//2, 0, grid_size//2, 0, 1, 0, 1)
    generate_leaves()

    trunk_vox = list(trunk_vox)
    rng.shuffle(trunk_vox)
    for i, (x, y, z) in enumerate(trunk_vox):
        voxels[x, y, z] = trunk_indices[i % len(trunk_indices)] if trunk_indices else 1

    leaf_vox = list(leaf_vox)
    rng.shuffle(leaf_vox)
    for i, (x, y, z) in enumerate(leaf_vox):
        if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
            if voxels[x, y, z] == 0:
                voxels[x, y, z] = leaf_indices[i % len(leaf_indices)] if leaf_indices else 1

    voxels = voxels.swapaxes(1, 2)
    # Clear any voxels outside the bounding box to ensure no invisible voxels
    coords = np.argwhere(voxels > 0)  # Ensure coords is defined
    if coords.size > 0:
        min_xyz = coords.min(axis=0).astype(int)
        max_xyz = coords.max(axis=0).astype(int)
        # Optimize clearing voxels outside the bounding box using NumPy slicing
        coords = np.argwhere(voxels > 0)  # Ensure coords is defined
        if coords.size > 0:
            min_xyz = coords.min(axis=0).astype(int)
            max_xyz = coords.max(axis=0).astype(int)
            mask = np.ones_like(voxels, dtype=bool)
            mask[min_xyz[0]:max_xyz[0]+1, min_xyz[1]:max_xyz[1]+1, min_xyz[2]:max_xyz[2]+1] = False
            voxels[mask] = 0
    return voxels, palette

# generate_pinegen_preview moved here
def generate_pinegen_preview(params, palette_name, grid_size=PREVIEW_GRID, progress_callback=None, cancel_check=None, view='front'):
    """Produce a preview image that matches exported output.

    Previously the preview scaled parameters and ran generation on a small grid,
    which could produce different structure than the full export. To ensure the
    preview always reflects the exported tree, generate the tree at full GRID
    resolution and then project & downscale to the requested preview size.
    """
    try:
        # Generate full-resolution voxels using the exact params so indices/colors
        # match the exporter (no parameter shrinking or approximations).
        vox, palette = generate_pinegen_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=progress_callback, cancel_check=cancel_check)

        # Project full-resolution voxels to an image using the same projection used for export
        img_full = project_voxels_to_image(vox, palette, GRID, view=view)

        # Downscale to the UI preview size (keep nearest neighbor to preserve palette indices)
        return img_full.resize((grid_size * 3, grid_size * 3), Image.NEAREST)
    except CancelledError:
        # propagate cancellation
        raise
    except Exception:
        # If full-generation fails (e.g., OOM on very constrained environments),
        # fall back to the previous faster path that generates on a reduced grid.
        shrink = grid_size / GRID
        params_preview = params.copy()
        params_preview['size'] *= shrink
        params_preview['trunkheight'] *= shrink
        vox, palette = generate_pinegen_tree(params_preview, palette_name, grid_size=grid_size, preview=True, progress_callback=progress_callback, cancel_check=cancel_check)
        img = project_voxels_to_image(vox, palette, grid_size, view=view)
        return img.resize((grid_size * 3, grid_size * 3), Image.NEAREST)

# export_pine moved here
# helper to orient voxels for export so MagicaVoxel front matches UI preview
def orient_voxels_for_export(voxels, view='front'):
    """Return reoriented copy of voxels matching preview->MagicaVoxel front mapping.
    'front' returns voxels unchanged. 'top' swaps X<->Z. Others unchanged.
    """
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


def export_pine(params, palette_name, prefix='pinegen', export_view='front'):
    voxels, palette = generate_pinegen_tree(params, palette_name, grid_size=GRID, preview=True)
    # Reorient voxels so exported file front matches preview front
    voxels_oriented = orient_voxels_for_export(voxels, view=export_view)
    exporter = VoxExporter(params, PINE_PALETTE_MAP, 'pine', 'pine', 'pinegen_counter.txt')
    loaded_palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name) if palette_name else (palette, [9,17], [57,65])
    return exporter.export(voxels_oriented, loaded_palette, leaf_indices, trunk_indices, prefix, preview=False)

# Define the missing `load_palette_png` function
def load_palette_png(filename):
    """Load a palette PNG file and return a list of RGBA tuples."""
    path = resource_path(filename)
    try:
        image = Image.open(path).convert("RGBA")
        pixels = list(image.getdata())
        if len(pixels) >= 256:
            return pixels[:256]
        return pixels + [(0, 0, 0, 0)] * (256 - len(pixels))
    except Exception:
        # Fallback grayscale palette
        return [(i, i, i, 255) for i in range(256)]