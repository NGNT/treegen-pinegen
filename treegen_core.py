"""
Core tree and pine generation logic extracted from the tkinter reference.
Provides generator functions and VoxExporter for both preview and export.
Added support for progress reporting and cooperative cancellation via callbacks.
"""
from PIL import Image
import numpy as np
import math
import random
import struct
import os

GRID = 256
PREVIEW_GRID = 64


def clamp(v, mi, ma):
    return max(mi, min(ma, v))

# resource_path for PyInstaller
def resource_path(filename):
    if hasattr(__import__('sys'), '_MEIPASS'):
        return os.path.join(__import__('sys')._MEIPASS, filename)
    return filename

from PIL import Image as PILImage


def load_palette_png(filename):
    path = resource_path(filename)
    image = PILImage.open(path).convert("RGBA")
    pixels = list(image.getdata())
    # If file not a single-row palette but larger, attempt to reduce to first 256 pixels
    if len(pixels) >= 256:
        return pixels[:256]
    # fallback grayscale palette
    return [(i, i, i, 255) for i in range(256)]


class CancelledError(Exception):
    pass

# VoxExporter ported from treegen-pinegen.py
class VoxExporter:
    def __init__(self, params, palette_map, palette_subdir, output_subdir, counter_file):
        self.params = params
        self.palette_map = palette_map
        self.palette_subdir = palette_subdir
        self.output_subdir = output_subdir
        self.counter_file = counter_file

    def load_palette(self, palette_name):
        key = os.path.basename(palette_name)
        path = resource_path(os.path.join('palettes', self.palette_subdir, key))
        palette = load_palette_png(path)
        # Construct default mapping if not found
        if isinstance(self.palette_map, dict):
            config = self.palette_map.get(key, next(iter(self.palette_map.values())))
        else:
            config = {'leaves': [9, 17], 'trunk': [57, 65]}
        return palette, config['leaves'], config['trunk']

    def export(self, voxels, palette, leaf_indices, trunk_indices, prefix, preview=False):
        if preview:
            return voxels, palette
        coords = np.argwhere(voxels > 0)
        if coords.size > 0:
            min_xyz = coords.min(axis=0)
            max_xyz = coords.max(axis=0)
            dims = max_xyz - min_xyz + 1
        else:
            min_xyz = np.array([0,0,0])
            dims = np.array([1,1,1])
        voxel_data = bytearray()
        for x, y, z in coords:
            c = int(voxels[x, y, z])
            x0, y0, z0 = (int(x - min_xyz[0]), int(y - min_xyz[1]), int(z - min_xyz[2]))
            voxel_data += struct.pack('<4B', x0, y0, z0, c)

        size_chunk = b'SIZE' + struct.pack('<ii', 12, 0)
        size_chunk += struct.pack('<iii', int(dims[0]), int(dims[1]), int(dims[2]))
        xyzi_payload = struct.pack('<i', len(voxel_data) // 4) + voxel_data
        xyzi_chunk = b'XYZI' + struct.pack('<ii', len(xyzi_payload), 0) + xyzi_payload
        rgba_payload = b''.join(struct.pack('<4B', *c) for c in palette)
        rgba_chunk = b'RGBA' + struct.pack('<ii', len(rgba_payload), 0) + rgba_payload
        main_content = size_chunk + xyzi_chunk + rgba_chunk
        main_chunk = b'MAIN' + struct.pack('<ii', 0, len(main_content)) + main_content
        vox_file = b'VOX ' + struct.pack('<i', 150) + main_chunk

        # counter handling
        if os.path.exists(self.counter_file):
            try:
                with open(self.counter_file, 'r') as f:
                    count = int(f.read().strip()) + 1
            except Exception:
                count = 1
        else:
            count = 1
        with open(self.counter_file, 'w') as f:
            f.write(str(count))

        out_dir = os.path.join('output', self.output_subdir)
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f'{prefix}_output{count}.vox')
        with open(filename, 'wb') as f:
            f.write(vox_file)
        return filename

# Import palette maps from original project if available
# Provide reasonable defaults if not
try:
    from treegen_packs import TREE_PALETTE_MAP, PINE_PALETTE_MAP
except Exception:
    # Fallback simple maps
    TREE_PALETTE_MAP = { 'tree_default.png': {'leaves':[9,17],'trunk':[57,65]} }
    PINE_PALETTE_MAP = { 'pine_default.png': {'leaves':[9,17],'trunk':[57,65]} }

# If the project doesn't have treegen_packs, attempt to load from treegen-pinegen.py
try:
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), 'treegen-pinegen.py')
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location('tk_treegen', path)
        tkmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tkmod)
        if hasattr(tkmod, 'TREE_PALETTE_MAP'):
            TREE_PALETTE_MAP = tkmod.TREE_PALETTE_MAP
        if hasattr(tkmod, 'PINE_PALETTE_MAP'):
            PINE_PALETTE_MAP = tkmod.PINE_PALETTE_MAP
except Exception:
    pass

# Ported generate_treegen_tree and generate_pinegen_tree (adapted and simplified for clarity)
# Both generators accept optional progress_callback(percent) and cancel_check() to support UI reporting and cancellation.

def generate_treegen_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=None, cancel_check=None):
    random.seed(int(params.get('seed', 1)))
    exporter = VoxExporter(params, TREE_PALETTE_MAP, 'tree', 'tree', 'treegen_counter.txt')
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
            if random.random() < get_branch_prob(i):
                b = 2
                var = get_branch_angle(i)
            for _ in range(b):
                dx2 = dx + random.uniform(-var, var)
                dy2 = dy + random.uniform(-var, var)
                dz2 = dz + random.uniform(-var, var)
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
                    d = random.randint(1, 6)
                    if d == 1: x2 -= 1
                    elif d == 2: x2 += 1
                    elif d in (3, 4):
                        z2 += 1 if random.uniform(-1, 1) < params.get('gravity', 0.0) else -1
                    elif d == 5: y2 -= 1
                    else: y2 += 1
        leaf_list = list(leaf_set)
        random.shuffle(leaf_list)
        for i, (x, y, z) in enumerate(leaf_list):
            idx = leaf_indices[i % len(leaf_indices)] if leaf_indices else 1
            if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
                if voxels[x, y, z] == 0:
                    voxels[x, y, z] = idx

    branches(grid_size//2, grid_size//2, 0, 0, 0, 1, 1)
    add_leaves()

    trunk_voxels = list(trunk_voxels)
    random.shuffle(trunk_voxels)
    for i, (x, y, z) in enumerate(trunk_voxels):
        idx = trunk_indices[i % len(trunk_indices)] if trunk_indices else 1
        voxels[x, y, z] = idx

    return voxels, palette


def generate_pinegen_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=None, cancel_check=None):
    random.seed(int(params.get('seed', 1)))
    exporter = VoxExporter(params, PINE_PALETTE_MAP, 'pine', 'pine', 'pinegen_counter.txt')
    palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name) if palette_name else ([(i,i,i,255) for i in range(256)], [9,17], [57,65])

    voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    trunk_vox = set()
    leaf_vox = set()
    gLeaves = []

    size = clamp(params.get('size', 1.0), 0.1, 3.0)
    twist_param = clamp(params.get('twisted', 0.5), 0.0, 4.0)
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
            dx += random.uniform(-1/steps, 1/steps)
            dy += random.uniform(-1/steps, 1/steps) + 0.4 / steps
            dz += random.uniform(-1/steps, 1/steps)
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
                a = random.uniform(0.0, math.tau)
                idx = math.cos(a)
                idy = random.uniform(0.5, 1.0) * branchdir
                idz = math.sin(a)
                idx, idy, idz = normalize(idx, idy, idz)
                il = (1.0 - i / max_iter) * branchlength * random.uniform(0.5, 1.5)
                t = random.uniform(0.0, 1.0)
                x2 = x + (x1 - x) * t
                y2 = y + (y1 - y) * t
                z2 = z + (z1 - z) * t
                branch(x2, y2, z2, idx, idy, idz, il + 3)

        if i < max_iter:
            t = i / max_iter
            var = twist_param * 0.5 * t * (1 - t)
            dx2 = dx + random.uniform(-var, var)
            dy2 = dy + random.uniform(-var, var)
            dz2 = dz + random.uniform(-var, var)
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
        sources = random.sample(gLeaves, min(num_clusters, len(gLeaves)))

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
                                if random.random() < 0.3:
                                    continue
                                lx, ly, lz = cx + dx, cy + dy, cz + dz
                                if 0 <= lx < grid_size and 0 <= ly < grid_size and 0 <= lz < grid_size:
                                    if voxels[lx, ly, lz] == 0:
                                        leaf_vox.add((lx, ly, lz))

    generate_branches(grid_size//2, 0, grid_size//2, 0, 1, 0, 1)
    generate_leaves()

    trunk_vox = list(trunk_vox)
    random.shuffle(trunk_vox)
    for i, (x, y, z) in enumerate(trunk_vox):
        voxels[x, y, z] = trunk_indices[i % len(trunk_indices)] if trunk_indices else 1

    leaf_vox = list(leaf_vox)
    random.shuffle(leaf_vox)
    for i, (x, y, z) in enumerate(leaf_vox):
        if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
            if voxels[x, y, z] == 0:
                voxels[x, y, z] = leaf_indices[i % len(leaf_indices)] if leaf_indices else 1

    voxels = voxels.swapaxes(1, 2)
    return voxels, palette


# Projection helpers

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


def generate_treegen_preview(params, palette_name, grid_size=PREVIEW_GRID, view='side', progress_callback=None, cancel_check=None):
    shrink = grid_size / GRID
    params_preview = params.copy()
    params_preview['size'] *= shrink
    params_preview['trunksize'] *= shrink
    vox, palette = generate_treegen_tree(params_preview, palette_name, grid_size=grid_size, preview=True, progress_callback=progress_callback, cancel_check=cancel_check)
    img = project_voxels_to_image(vox, palette, grid_size, view=view)
    return img.resize((grid_size*3, grid_size*3), Image.NEAREST)


def generate_pinegen_preview(params, palette_name, grid_size=PREVIEW_GRID, progress_callback=None, cancel_check=None):
    shrink = grid_size / GRID
    params_preview = params.copy()
    params_preview['size'] *= shrink
    params_preview['trunkheight'] *= shrink
    vox, palette = generate_pinegen_tree(params_preview, palette_name, grid_size=grid_size, preview=True, progress_callback=progress_callback, cancel_check=cancel_check)
    proj = vox.max(axis=0).T[::-1, :]
    img_arr = np.zeros((grid_size, grid_size, 4), np.uint8)
    for idx, rgba in enumerate(palette):
        arr = np.array(rgba, dtype=np.uint8)
        img_arr[proj == idx] = arr
    return Image.fromarray(img_arr, 'RGBA').resize((grid_size*3, grid_size*3), Image.NEAREST)

# Export helper wrappers for use by external processes/UI
def export_tree(params, palette_name, prefix='treegen'):
    """Generate tree voxels and export to .vox using VoxExporter. Returns filename."""
    voxels, palette = generate_treegen_tree(params, palette_name, grid_size=GRID, preview=True)
    exporter = VoxExporter(params, TREE_PALETTE_MAP, 'tree', 'tree', 'treegen_counter.txt')
    # ensure palette and indices match exporter's mapping
    loaded_palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name) if palette_name else (palette, [9,17], [57,65])
    # If loaded_palette differs, use loaded_palette for file metadata
    return exporter.export(voxels, loaded_palette, leaf_indices, trunk_indices, prefix, preview=False)


def export_pine(params, palette_name, prefix='pinegen'):
    """Generate pine voxels and export to .vox using VoxExporter. Returns filename."""
    voxels, palette = generate_pinegen_tree(params, palette_name, grid_size=GRID, preview=True)
    exporter = VoxExporter(params, PINE_PALETTE_MAP, 'pine', 'pine', 'pinegen_counter.txt')
    loaded_palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name) if palette_name else (palette, [9,17], [57,65])
    return exporter.export(voxels, loaded_palette, leaf_indices, trunk_indices, prefix, preview=False)
