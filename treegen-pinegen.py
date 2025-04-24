import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import math
import struct
import random
import subprocess
import platform
import os
import sys
from dataclasses import dataclass, asdict

GRID = 256
PREVIEW_GRID = 64

def clamp(v, mi, ma):
    return max(mi, min(ma, v))

def resource_path(filename):
    """Get path to resource for PyInstaller --onefile"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return filename

def load_palette_png(filename):
    path = resource_path(filename)
    image = Image.open(path).convert("RGBA")
    pixels = list(image.getdata())
    if len(pixels) != 256:
        raise ValueError("Palette must be exactly 256 pixels wide")
    return pixels

# === TREEGEN palette index map ===
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

# === PINEGEN palette index map ===
PINE_PALETTE_MAP = {
    "pine_default.png": {"leaves": [9, 17], "trunk": [57, 65]},
    "pine_basic.png":   {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))},
    "redpine.png":      {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))},
    "pine_sapling.png": {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))},
    "scotspine.png":    {"leaves": list(range(9, 17)), "trunk": list(range(57, 65))}
}

@dataclass
class TreeGenParams:
    size: float = 1.0
    trunksize: float = 1.0
    spread: float = 0.5
    twisted: float = 0.5
    leaves: float = 1.0
    gravity: float = 0.0
    iterations: int = 12
    wide: float = 0.5
    seed: int = 1

    def __post_init__(self):
        errors = []
        if not (0.1 <= self.size <= 3.0):
            errors.append(f"size {self.size} out of range [0.1, 3.0]")
        if not (0.1 <= self.trunksize <= 3.0):
            errors.append(f"trunksize {self.trunksize} out of range [0.1, 3.0]")
        if not (0.0 <= self.spread <= 1.0):
            errors.append(f"spread {self.spread} out of range [0.0, 1.0]")
        if not (0.0 <= self.twisted <= 1.0):
            errors.append(f"twisted {self.twisted} out of range [0.0, 1.0]")
        if not (0.0 <= self.leaves <= 3.0):
            errors.append(f"leaves {self.leaves} out of range [0.0, 3.0]")
        if not (-1.0 <= self.gravity <= 1.0):
            errors.append(f"gravity {self.gravity} out of range [-1.0, 1.0]")
        if not (5 <= self.iterations <= 15):
            errors.append(f"iterations {self.iterations} out of range [5, 15]")
        if not (0.0 <= self.wide <= 1.0):
            errors.append(f"wide {self.wide} out of range [0.0, 1.0]")
        if not (1 <= self.seed <= 9999):
            errors.append(f"seed {self.seed} out of range [1, 9999]")
        if errors:
            raise ValueError("; ".join(errors))

@dataclass
class PineGenParams:
    size: float = 1.0
    twisted: float = 0.5
    trunksize: float = 2.0
    trunkheight: float = 1.0
    branchdensity: float = 1.0
    branchlength: float = 1.0
    branchdir: float = -0.5
    leaves: float = 1.0
    leaf_radius: float = 2.0
    leaf_stretch: float = 1.5
    leaf_bias: float = -0.3
    seed: int = 1

    def __post_init__(self):
        errors = []
        if not (0.1 <= self.size <= 3.0):
            errors.append(f"size {self.size} out of range [0.1, 3.0]")
        if not (0.0 <= self.twisted <= 3.0):
            errors.append(f"twisted {self.twisted} out of range [0.0, 3.0]")
        if not (1.0 <= self.trunksize <= 3.0):
            errors.append(f"trunksize {self.trunksize} out of range [1.0, 3.0]")
        if not (0.0 <= self.trunkheight <= 5.0):
            errors.append(f"trunkheight {self.trunkheight} out of range [0.0, 5.0]")
        if not (0.0 <= self.branchdensity <= 3.0):
            errors.append(f"branchdensity {self.branchdensity} out of range [0.0, 3.0]")
        if not (0.0 <= self.branchlength <= 3.0):
            errors.append(f"branchlength {self.branchlength} out of range [0.0, 3.0]")
        if not (-5.0 <= self.branchdir <= 5.0):
            errors.append(f"branchdir {self.branchdir} out of range [-5.0, 5.0]")
        if not (0.0 <= self.leaves <= 2.0):
            errors.append(f"leaves {self.leaves} out of range [0.0, 2.0]")
        if not (1.0 <= self.leaf_radius <= 4.0):
            errors.append(f"leaf_radius {self.leaf_radius} out of range [1.0, 4.0]")
        if not (0.5 <= self.leaf_stretch <= 3.0):
            errors.append(f"leaf_stretch {self.leaf_stretch} out of range [0.5, 3.0]")
        if not (-1.0 <= self.leaf_bias <= 1.0):
            errors.append(f"leaf_bias {self.leaf_bias} out of range [-1.0, 1.0]")
        if not (1 <= self.seed <= 9999):
            errors.append(f"seed {self.seed} out of range [1, 9999]")
        if errors:
            raise ValueError("; ".join(errors))

# Shared exporter for building and writing .vox files
class VoxExporter:
    def __init__(self, params, palette_map, palette_subdir, output_subdir, counter_file):
        self.params = params
        self.palette_map = palette_map
        self.palette_subdir = palette_subdir
        self.output_subdir = output_subdir
        self.counter_file = counter_file

    def load_palette(self, palette_name):
        # accept full path or filename, but always use basename for lookup
        key = os.path.basename(palette_name)
        path = resource_path(os.path.join("palettes", self.palette_subdir, key))
        palette = load_palette_png(path)
        config = self.palette_map.get(key, next(iter(self.palette_map.values())))
        return palette, config["leaves"], config["trunk"]

    def export(self, voxels, palette, leaf_indices, trunk_indices, prefix):
        # compute bounding box of non-zero voxels
        coords = np.argwhere(voxels > 0)
        if coords.size > 0:
            min_xyz = coords.min(axis=0)
            max_xyz = coords.max(axis=0)
            dims = max_xyz - min_xyz + 1
        else:
            # empty model: single voxel on origin
            min_xyz = np.array([0, 0, 0])
            dims = np.array([1, 1, 1])
        # pack voxel data with shifted positions
        voxel_data = bytearray()
        for x, y, z in coords:
            c = voxels[x, y, z]
            x0, y0, z0 = (x - int(min_xyz[0]), y - int(min_xyz[1]), z - int(min_xyz[2]))
            voxel_data += struct.pack('<4B', x0, y0, z0, c)

        # SIZE chunk with dynamic dimensions
        size_chunk = b'SIZE' + struct.pack('<ii', 12, 0)
        size_chunk += struct.pack('<iii', dims[0], dims[1], dims[2])
        # XYZI chunk
        xyzi_payload = struct.pack('<i', len(voxel_data) // 4) + voxel_data
        xyzi_chunk = b'XYZI' + struct.pack('<ii', len(xyzi_payload), 0) + xyzi_payload
        # RGBA chunk
        rgba_payload = b''.join(struct.pack('<4B', *c) for c in palette)
        rgba_chunk = b'RGBA' + struct.pack('<ii', len(rgba_payload), 0) + rgba_payload

        # MAIN chunk assembly
        main_content = size_chunk + xyzi_chunk + rgba_chunk
        main_chunk = b'MAIN' + struct.pack('<ii', 0, len(main_content)) + main_content
        vox_file = b'VOX ' + struct.pack('<i', 150) + main_chunk

        # counter file handling
        if os.path.exists(self.counter_file):
            try:
                with open(self.counter_file, 'r') as f:
                    count = int(f.read().strip()) + 1
            except:
                count = 1
        else:
            count = 1
        with open(self.counter_file, 'w') as f:
            f.write(str(count))

        # ensure output directory exists
        out_dir = os.path.join('output', self.output_subdir)
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f'{prefix}_output{count}.vox')
        with open(filename, 'wb') as f:
            f.write(vox_file)
        return filename

def generate_treegen_tree(params, palette_name):
    random.seed(params['seed'])
    # initialize shared exporter and load palette/index mappings
    exporter = VoxExporter(params, TREE_PALETTE_MAP, "tree", "tree", "treegen_counter.txt")
    palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name)

    voxels = np.zeros((GRID, GRID, GRID), dtype=np.uint8)
    gLeaves = []

    # avoid division by zero on iterations
    iters = max(params.get('iterations', 1), 1)
    size = 150 * params['size'] / iters
    gTrunkSize = params['trunksize'] * params['size'] * 6
    wide = min(params['wide'], 0.95)
    gBranchLength0 = size * (1 - wide)
    gBranchLength1 = size * wide

    def normalize(x, y, z):
        l = math.sqrt(x*x + y*y + z*z)
        return (x/l, y/l, z/l) if l > 0 else (0, 0, 1)

    trunk_voxels = []

    def draw_line(x0, y0, z0, x1, y1, z1, r0, r1):
        steps = int(math.dist([x0, y0, z0], [x1, y1, z1]) * 2)
        if steps == 0:
            steps = 1
        for i in range(steps + 1):
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
                            if 0 <= vx < GRID and 0 <= vy < GRID and 0 <= vz < GRID:
                                trunk_voxels.append((vx, vy, vz))

    def get_branch_length(i):
        t = math.sqrt((i - 1) / iters)
        return gBranchLength0 + t * (gBranchLength1 - gBranchLength0)

    def get_branch_size(i):
        t = math.sqrt((i - 1) / iters)
        return (1 - t) * gTrunkSize

    def get_branch_angle(i):
        t = math.sqrt((i - 1) / iters)
        return 2.0 * params['spread'] * t

    def get_branch_prob(i):
        return math.sqrt((i - 1) / iters)

    def branches(x, y, z, dx, dy, dz, i):
        l = get_branch_length(i)
        s0 = get_branch_size(i)
        s1 = get_branch_size(i+1)
        x1 = x + dx * l
        y1 = y + dy * l
        z1 = z + dz * l
        draw_line(x, y, z, x1, y1, z1, s0, s1)

        if i < iters:
            b = 1
            var = i * 0.2 * params['twisted']
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

    leaf_voxels = []

    def add_leaves():
        for pos in gLeaves:
            x1, y1, z1 = map(int, pos)
            for _ in range(int(5 * params['leaves'])):
                x2, y2, z2 = x1, y1, z1
                for _ in range(int(50 * params['leaves'])):
                    leaf_voxels.append((x2, y2, z2))
                    d = random.randint(1, 6)
                    if d == 1: x2 -= 1
                    elif d == 2: x2 += 1
                    elif d in (3, 4):
                        z2 += 1 if random.uniform(-1, 1) < params['gravity'] else -1
                    elif d == 5: y2 -= 1
                    else: y2 += 1

        random.shuffle(leaf_voxels)
        for i, (x, y, z) in enumerate(leaf_voxels):
            idx = leaf_indices[i % len(leaf_indices)]
            if 0 <= x < GRID and 0 <= y < GRID and 0 <= z < GRID:
                if voxels[x, y, z] == 0:
                    voxels[x, y, z] = idx

    branches(GRID//2, GRID//2, 0, 0, 0, 1, 1)
    add_leaves()

    random.shuffle(trunk_voxels)
    for i, (x, y, z) in enumerate(trunk_voxels):
        idx = trunk_indices[i % len(trunk_indices)]
        voxels[x, y, z] = idx

    # export voxel data via shared exporter
    return exporter.export(voxels, palette, leaf_indices, trunk_indices, "treegen")
    
def generate_pinegen_tree(params, palette_name):
    random.seed(int(params["seed"]))
    # initialize shared exporter and load palette/index mappings
    exporter = VoxExporter(params, PINE_PALETTE_MAP, "pine", "pine", "pinegen_counter.txt")
    palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name)
    voxels = np.zeros((GRID, GRID, GRID), dtype=np.uint8)
    trunk_vox, leaf_vox = [], []
    gLeaves = []

    # Parameters
    size = clamp(params["size"], 0.1, 3.0)
    twist_param = clamp(params["twisted"], 0.0, 3.0)
    trunkheight = params["trunkheight"] * 10
    density = clamp(params["branchdensity"], 0, 3) * 30
    branchlength = clamp(params["branchlength"], 0, 3) * size * 20
    branchdir = clamp(params["branchdir"], -5, 5)
    leaves = clamp(params["leaves"], 0, 2)
    trunk_width = size * params.get("trunksize", 2) * 1.1  # slightly amplify trunk size strength
    max_iter = math.floor(100 * size / 5)
    fixed_size = 5

    def draw_line(x0, y0, z0, x1, y1, z1, r):
        steps = int(math.dist([x0, y0, z0], [x1, y1, z1]) * 2)
        if steps == 0:
            steps = 1
        for i in range(steps + 1):
            t = i / steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            z = z0 + t * (z1 - z0)
            for dx in range(-math.ceil(r), math.ceil(r)+1):
                for dy in range(-math.ceil(r), math.ceil(r)+1):
                    for dz in range(-math.ceil(r), math.ceil(r)+1):
                        if dx*dx + dy*dy + dz*dz <= r*r:
                            xi, yi, zi = int(x+dx), int(y+dy), int(z+dz)
                            if 0 <= xi < GRID and 0 <= yi < GRID and 0 <= zi < GRID:
                                trunk_vox.append((xi, yi, zi))

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
            # apply half-strength bell-curve for gentler bend
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
        radius = int(clamp(params.get("leaf_radius", 2), 1, 4))
        vertical_stretch = clamp(params.get("leaf_stretch", 1.5), 0.1, 5.0)
        direction_bias = clamp(params.get("leaf_bias", -0.3), -1.0, 1.0)
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
                                if 0 <= lx < GRID and 0 <= ly < GRID and 0 <= lz < GRID:
                                    if voxels[lx, ly, lz] == 0:
                                        leaf_vox.append((lx, ly, lz))

    generate_branches(GRID//2, 0, GRID//2, 0, 1, 0, 1)
    generate_leaves()

    random.shuffle(trunk_vox)
    for i, (x, y, z) in enumerate(trunk_vox):
        voxels[x, y, z] = trunk_indices[i % len(trunk_indices)]

    random.shuffle(leaf_vox)
    for i, (x, y, z) in enumerate(leaf_vox):
        if 0 <= x < GRID and 0 <= y < GRID and 0 <= z < GRID:
            if voxels[x, y, z] == 0:
                voxels[x, y, z] = leaf_indices[i % len(leaf_indices)]

    # rotate axes: pine built along y-axis but exporter expects z-axis vertical
    voxels = voxels.swapaxes(1, 2)
    # export via shared exporter
    return exporter.export(voxels, palette, leaf_indices, trunk_indices, "pinegen")

# Preview image generator (low-res)
def generate_treegen_preview(params, palette_name, grid_size=PREVIEW_GRID, view="side"):
    global GRID
    orig_GRID = GRID
    GRID = grid_size
    orig_export = VoxExporter.export
    def _preview_export(self, voxels, palette, leaf_indices, trunk_indices, prefix):
        return voxels, palette
    VoxExporter.export = _preview_export
    try:
        # shrink model to fit preview grid
        shrink = grid_size / orig_GRID
        params_preview = params.copy()
        params_preview['size'] *= shrink
        params_preview['trunksize'] *= shrink
        vox, palette = generate_treegen_tree(params_preview, palette_name)
    finally:
        GRID = orig_GRID
        VoxExporter.export = orig_export
    # choose projection based on view and right-side-up
    if view == "top":
        proj = vox.max(axis=2)
    elif view == "front":
        proj = vox.max(axis=1).T[::-1, :]
    elif view == "side":
        proj = vox.max(axis=0).T[::-1, :]
    else:
        # fallback to top
        proj = vox.max(axis=2)
    img_arr = np.zeros((grid_size, grid_size, 4), np.uint8)
    for idx, rgba in enumerate(palette):
        img_arr[proj == idx] = rgba
    img = Image.fromarray(img_arr, 'RGBA').resize((grid_size*4, grid_size*4), Image.NEAREST)
    return img

# Preview image generator for Pine (low-res)
def generate_pinegen_preview(params, palette_name, grid_size=PREVIEW_GRID):
    global GRID
    orig_GRID = GRID
    GRID = grid_size
    orig_export = VoxExporter.export
    def _preview_export(self, voxels, palette, leaf_indices, trunk_indices, prefix):
        return voxels, palette
    VoxExporter.export = _preview_export
    try:
        # shrink to fit preview
        shrink = grid_size / orig_GRID
        params_preview = params.copy()
        params_preview['size'] *= shrink
        # remove extra shrink on trunk size so thickness varies in preview
        # params_preview['trunksize'] *= shrink
        params_preview['trunkheight'] *= shrink
        vox, palette = generate_pinegen_tree(params_preview, palette_name)
    finally:
        GRID = orig_GRID
        VoxExporter.export = orig_export
    # side projection (Y horizontal, Z vertical)
    proj = vox.max(axis=0).T[::-1, :]
    img_arr = np.zeros((grid_size, grid_size, 4), np.uint8)
    for idx, rgba in enumerate(palette):
        img_arr[proj == idx] = rgba
    img = Image.fromarray(img_arr, 'RGBA').resize((grid_size*4, grid_size*4), Image.NEAREST)
    return img

# Helper to add a slider row with label, reset button, and tooltip
def add_slider(parent, label, var, mn, mx, default, tooltip, status_var):
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=4)
    ttk.Label(row, text=label).pack(side="left", padx=(0, 5))
    val_text = f"{var.get():.2f}" if isinstance(var.get(), float) else str(var.get())
    val_label = ttk.Label(row, text=val_text)
    val_label.pack(side="right")
    def make_callback(v=var, lbl=val_label):
        def update_val(_):
            lbl.config(text=f"{v.get():.2f}" if isinstance(v.get(), float) else str(v.get()))
        return update_val
    def make_reset(v=var, default=default, lbl=val_label):
        def reset():
            v.set(default)
            lbl.config(text=f"{v.get():.2f}" if isinstance(v.get(), float) else str(v.get()))
        return reset
    ttk.Button(row, text="⭯", width=3, command=make_reset()).pack(side="right", padx=5)
    ttk.Scale(row, from_=mn, to=mx, variable=var, orient="horizontal", command=make_callback()).pack(fill="x", padx=5)
    if tooltip is not None:
        row.bind("<Enter>", lambda e: status_var.set(tooltip))
        row.bind("<Leave>", lambda e: status_var.set("Ready"))
    # update displayed value whenever the variable changes
    var.trace_add('write', lambda *args, v=var, lbl=val_label: lbl.config(text=f"{v.get():.2f}" if isinstance(v.get(), float) else str(v.get())))
    return row

def build_treegen_gui(tab):
    # Treegen branding
    tree_brand_img = ImageTk.PhotoImage(Image.open(resource_path("treegen_brand.png")))
    ttk.Label(tab, image=tree_brand_img).pack(pady=(10,10))
    tab.tree_brand_img = tree_brand_img
    # embedded preview in the tab
    preview_label_tree = ttk.Label(tab)
    preview_label_tree.pack(side="right", anchor="n", padx=(20,0), pady=(20,10))

    ttk.Label(tab, text="Treegen v1.4.1 🌳 by NGNT", font=("Arial", 14, "bold")).pack()

    # === Palette dropdown
    tree_palette_path = resource_path("palettes/tree")
    palette_files = [f for f in os.listdir(tree_palette_path) if f.endswith(".png")]
    palette_var = tk.StringVar(value=palette_files[0])
    palette_row = ttk.Frame(tab)
    palette_row.pack(fill="x", pady=4)
    ttk.Label(palette_row, text="Palette").pack(side="left", padx=(0, 5))
    palette_dropdown = ttk.Combobox(palette_row, textvariable=palette_var, values=palette_files, state="readonly")
    palette_dropdown.pack(fill="x", expand=True)

    # === Sliders
    controls = {
        "size":        tk.DoubleVar(value=1.0),
        "trunksize":   tk.DoubleVar(value=1.0),
        "spread":      tk.DoubleVar(value=0.5),
        "twisted":     tk.DoubleVar(value=0.5),
        "leaves":      tk.DoubleVar(value=1.0),
        "gravity":     tk.DoubleVar(value=0.0),
        "iterations":  tk.IntVar(value=12),
        "wide":        tk.DoubleVar(value=0.5),
        "seed":        tk.IntVar(value=1),
        "open_after":  tk.BooleanVar(value=True),
        "status":      tk.StringVar(value="Ready")
    }

    slider_defs = [
        ("Size", controls["size"], 0.1, 3.0),
        ("Trunk Size", controls["trunksize"], 0.1, 1.2),
        ("Spread", controls["spread"], 0.0, 1.0),
        ("Twist", controls["twisted"], 0.0, 1.0),
        ("Leafiness", controls["leaves"], 0.0, 3.0),
        ("Gravity", controls["gravity"], -1.0, 1.0),
        ("Iterations", controls["iterations"], 5, 15),
        ("Wide", controls["wide"], 0.0, 1.0),
        ("Seed", controls["seed"], 1, 9999)
    ]

    tree_tooltips = {
        "Size": "Controls overall height/scale of the tree",
        "Trunk Size": "How thick the trunk appears",
        "Spread": "How much branches spread out sideways",
        "Twist": "How twisted or chaotic the branches are",
        "Leafiness": "Controls how many leaves spawn",
        "Gravity": "Positive pulls leaves up, negative pulls down",
        "Iterations": "How complex/tall the tree structure is",
        "Wide": "Controls base vs top branch bias (0 = base, 1 = top)",
        "Seed": "Seed for randomness — same number = same tree"
    }

    defaults = {label: var.get() for label, var, *_ in slider_defs}

    for label, var, mn, mx in slider_defs:
        add_slider(tab, label, var, mn, mx, defaults[label], tree_tooltips.get(label, ""), controls["status"])

    ttk.Checkbutton(tab, text="Open file after generation", variable=controls["open_after"]).pack(pady=(5, 0))

    def update_preview(*args):
        # fetch parameters
        raw = {k: v.get() for k, v in controls.items() if k not in ('status', 'open_after')}
        params = TreeGenParams(**raw)
        # generate preview image at grid_size=128
        img = generate_treegen_preview(asdict(params), palette_var.get(), grid_size=PREVIEW_GRID)
        img_tk = ImageTk.PhotoImage(img)
        preview_label_tree.config(image=img_tk)
        preview_label_tree.image = img_tk
        controls['status'].set('Ready')

    for var in controls.values():
        if isinstance(var, (tk.DoubleVar, tk.IntVar)):
            var.trace_add('write', update_preview)
    palette_var.trace_add('write', update_preview)

    # initial preview
    update_preview()

    # Randomize sliders
    def randomize_sliders():
        for _, var, mn, mx in slider_defs:
            if isinstance(var, tk.DoubleVar):
                var.set(random.uniform(mn, mx))
            elif isinstance(var, tk.IntVar):
                var.set(random.randint(int(mn), int(mx)))
        update_preview()
    def generate():
        try:
            # Collect and validate parameters via dataclass
            raw = {k: v.get() for k, v in controls.items() if k not in ("status", "open_after")}
            params_obj = TreeGenParams(**raw)
            filename = generate_treegen_tree(asdict(params_obj), palette_var.get())
            controls["status"].set(f"✅ Generated {filename}!")
            if controls["open_after"].get():
                if platform.system() == "Windows":
                    os.startfile(filename)
                elif platform.system() == "Darwin":
                    subprocess.call(["open", filename])
                else:
                    subprocess.call(["xdg-open", filename])
        except Exception as e:
            messagebox.showerror("Error", str(e))
            controls["status"].set("⚠️ Generation failed")
    # group Randomize and Generate side by side
    btn_frame = ttk.Frame(tab)
    btn_frame.pack(pady=10)
    ttk.Button(btn_frame, text="🔀 Randomize", command=randomize_sliders).pack(side="left", padx=(0,5))
    ttk.Button(btn_frame, text="🌳 Generate Tree", command=generate).pack(side="left")
    ttk.Label(tab, textvariable=controls["status"]).pack(pady=5)

    return controls
    
def build_pinegen_gui(tab):
    # Pinegen branding
    pine_brand_img = ImageTk.PhotoImage(Image.open(resource_path("pinegen_brand.png")))
    ttk.Label(tab, image=pine_brand_img).pack(pady=(10,10))
    tab.pine_brand_img = pine_brand_img
    # embedded preview in the tab
    preview_label_pine = ttk.Label(tab)
    preview_label_pine.pack(side="right", anchor="n", padx=(20,0), pady=(20,10))

    ttk.Label(tab, text="Pinegen v1.4.1 🌲 by NGNT", font=("Arial", 14, "bold")).pack()

    pine_palette_path = resource_path("palettes/pine")
    palette_files = [f for f in os.listdir(pine_palette_path) if f.endswith(".png")]
    palette_var = tk.StringVar(value=palette_files[0])
    palette_row = ttk.Frame(tab)
    palette_row.pack(fill="x", pady=4)
    ttk.Label(palette_row, text="Palette").pack(side="left", padx=(0, 5))
    palette_dropdown = ttk.Combobox(palette_row, textvariable=palette_var, values=palette_files, state="readonly")
    palette_dropdown.pack(fill="x", expand=True)

    controls = {
        "size":         tk.DoubleVar(value=1.0),
        "twisted":      tk.DoubleVar(value=0.5),
        "trunksize":    tk.DoubleVar(value=2.0),
        "trunkheight":  tk.DoubleVar(value=1.0),
        "branchdensity":tk.DoubleVar(value=1.0),
        "branchlength": tk.DoubleVar(value=1.0),
        "branchdir":    tk.DoubleVar(value=-0.5),
        "leaves":       tk.DoubleVar(value=1.0),
        "leaf_radius":  tk.DoubleVar(value=2.0),
        "leaf_stretch": tk.DoubleVar(value=1.5),
        "leaf_bias":    tk.DoubleVar(value=-0.3),
        "seed":         tk.IntVar(value=1),
        "open_after":   tk.BooleanVar(value=True),
        "status":       tk.StringVar(value="Ready")
    }

    slider_defs = [
        ("Size", controls["size"], 0.1, 3.0),
        ("Twist", controls["twisted"], 0.0, 3.0),
        ("Trunk Size", controls["trunksize"], 1.0, 3.0),
        ("Trunk Height", controls["trunkheight"], 0.0, 5.0),
        ("Branch Density", controls["branchdensity"], 0.0, 3.0),
        ("Branch Length", controls["branchlength"], 0.0, 3.0),
        ("Branch Direction", controls["branchdir"], -5.0, 5.0),
        ("Leafiness", controls["leaves"], 0.0, 2.0),
        ("Leaf Radius", controls["leaf_radius"], 1.0, 4.0),
        ("Leaf Stretch", controls["leaf_stretch"], 0.5, 3.0),
        ("Leaf Bias", controls["leaf_bias"], -1.0, 1.0),
        ("Seed", controls["seed"], 1, 9999)
    ]
    
    pine_tooltips = {
        "Size": "Overall size/scale of the pine tree",
        "Twist": "How wiggly the trunk is",
        "Trunk Size": "Scales the width of the trunk",
        "Trunk Height": "Height before branches appear",
        "Branch Density": "How many side branches",
        "Branch Length": "How far the branches grow",
        "Branch Direction": "Tilt of the branches",
        "Leafiness": "How much leaf coverage",
        "Leaf Radius": "How far leaves spread outward from branch ends",
        "Leaf Stretch": "Vertical stretch of leaf clusters (cone/tube shape)",
        "Leaf Bias": "Controls leaf cluster tilt: -1 = downward, +1 = upward", 
        "Seed": "Random seed for variation"
    }

    defaults = {label: var.get() for label, var, *_ in slider_defs}

    for label, var, mn, mx in slider_defs:
        add_slider(tab, label, var, mn, mx, defaults[label], pine_tooltips.get(label, ""), controls["status"])

    ttk.Checkbutton(tab, text="Open file after generation", variable=controls["open_after"]).pack(pady=(5, 0))

    def update_pine_preview(*args):
        raw = {k: v.get() for k, v in controls.items() if k not in ('status', 'open_after')}
        params_obj = PineGenParams(**raw)
        img = generate_pinegen_preview(asdict(params_obj), palette_var.get(), grid_size=PREVIEW_GRID)
        img_tk = ImageTk.PhotoImage(img)
        preview_label_pine.config(image=img_tk)
        preview_label_pine.image = img_tk
        controls['status'].set('Ready')

    for var in controls.values():
        if isinstance(var, (tk.DoubleVar, tk.IntVar)):
            var.trace_add('write', update_pine_preview)
    palette_var.trace_add('write', update_pine_preview)
    # initial pine preview
    update_pine_preview()

    # Randomize sliders
    def randomize_sliders():
        for _, var, mn, mx in slider_defs:
            if isinstance(var, tk.DoubleVar):
                var.set(random.uniform(mn, mx))
            elif isinstance(var, tk.IntVar):
                var.set(random.randint(int(mn), int(mx)))
        update_pine_preview()
    def generate():
        try:
            # Collect and validate parameters via dataclass
            raw = {k: v.get() for k, v in controls.items() if k not in ("status", "open_after")}
            params_obj = PineGenParams(**raw)
            filename = generate_pinegen_tree(asdict(params_obj), palette_var.get())
            controls["status"].set(f"✅ Generated {filename}!")
            if controls["open_after"].get():
                if platform.system() == "Windows":
                    os.startfile(filename)
                elif platform.system() == "Darwin":
                    subprocess.call(["open", filename])
                else:
                    subprocess.call(["xdg-open", filename])
        except Exception as e:
            messagebox.showerror("Error", str(e))
            controls["status"].set("⚠️ Generation failed")
    # group Randomize and Generate side by side
    btn_frame = ttk.Frame(tab)
    btn_frame.pack(pady=10)
    ttk.Button(btn_frame, text="🔀 Randomize", command=randomize_sliders).pack(side="left", padx=(0,5))
    ttk.Button(btn_frame, text="🌲 Generate Pine Tree", command=generate).pack(side="left")
    ttk.Label(tab, textvariable=controls["status"]).pack(pady=5)

    return controls


# === MAIN ENTRYPOINT ===
def run_gui():
    root = tk.Tk()
    root.title("Voxel Tree Generator Studio")
    root.geometry("800x950")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    tree_tab = ttk.Frame(notebook)
    pine_tab = ttk.Frame(notebook)

    notebook.add(tree_tab, text="Treegen 🌳")
    notebook.add(pine_tab, text="Pinegen 🌲")

    build_treegen_gui(tree_tab)
    build_pinegen_gui(pine_tab)

    root.mainloop()


if __name__ == "__main__":
    run_gui()