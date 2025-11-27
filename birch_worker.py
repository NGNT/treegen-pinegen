# (full file content with local-RNG changes)
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
    get_internal_palette = None
    _palette_manager = None

try:
    import numpy as np
except Exception:
    raise SystemExit('numpy is required for birch_worker')

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


def load_palette_png(filename):
    # try opening palette relative to project resources
    path = resource_path(filename)
    try:
        image = PILImage.open(path).convert("RGBA")
        pixels = list(image.getdata())
        if len(pixels) >= 256:
            return pixels[:256]
        return pixels + [(0, 0, 0, 0)] * (256 - len(pixels))
    except Exception:
        # fallback grayscale palette
        return [(i, i, i, 255) for i in range(256)]


class VoxExporter:
    def __init__(self, params, palette_map=None, palette_subdir='tree', output_subdir='tree', counter_file='birch_counter.txt'):
        self.params = params
        self.palette_map = palette_map or {'default': {'leaves':[9,17],'trunk':[57,65]}}
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

    def export(self, voxels, palette, leaf_indices, trunk_indices, prefix='birch', preview=False):
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


class CancelledError(Exception):
    pass


# Ported generation logic (adapted from project core)
def generate_birchgen_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=None, cancel_check=None):
    # Use local RNG so we don't reseed the global generator
    seed = int(params.get('seed', 1))
    rng = random.Random(seed)

    exporter = VoxExporter(params)
    palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name) if palette_name else ([(i,i,i,255) for i in range(256)], [9,17], [57,65])

    voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    gLeaves = []

    iters = max(int(params.get('iterations', 1)), 1)
    size = 150 * params.get('size', 1.0) / iters
    # Use a smaller trunk thickness multiplier for birch to keep trunks slender.
    # Scale by a reduced multiplier (3) rather than the Treegen core value to avoid overly thick trunks.
    gTrunkSize = (params.get('trunksize', 1.0) * 0.7 + 0.3) * params.get('size', 1.0) * 3
    wide = min(params.get('wide', 0.5), 0.95)
    spread = clamp(params.get('spread', 0.4), 0.3, 0.8)  # moderate spread
    gBranchLength0 = size * (1 - wide * 0.5)
    gBranchLength1 = size * (wide * 0.5 + spread * 0.1)  # add slight spread influence

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
        return 2.0 * params.get('spread', 0.5) * 0.5 * t

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
            try:
                progress_callback(min(progress_done / progress_total, 1.0))
            except Exception:
                pass

    def add_leaves():
        leaf_set = set()
        # For birch, create more numerous but smaller leaf clusters with a slight downward bias
        # Only create leaves around branch tips and avoid positions too close to trunk base.
        min_tip_height = trunk_height * 0.25 if 'trunk_height' in locals() else 4
        trunk_center_x = grid_size // 2
        for pos in gLeaves:
            # use float positions as sources but convert to int for placement
            x1_f, y1_f, z1_f = pos
            x1, y1, z1 = int(round(x1_f)), int(round(y1_f)), int(round(z1_f))
            # skip tips that are too low (near trunk base)
            if z1 < min_tip_height:
                continue
            # skip tips that are essentially on the trunk column
            if (x1 - trunk_center_x)**2 + (y1 - trunk_center_x)**2 <= (trunk_radius + 1)**2:
                continue
            clusters = max(1, int(2 * params.get('leaves', 1.0)))
            for _c in range(clusters):
                # start near the leaf source and perform a short random walk to populate a small cluster
                x2 = x1 + rng.randint(-1, 1)
                y2 = y1 + rng.randint(-1, 1)
                z2 = z1 + rng.randint(-1, 1)
                steps = max(2, int(8 * params.get('leaves', 1.0)))
                for _ in range(steps):
                    # avoid placing leaves inside the trunk radius
                    if (x2 - trunk_center_x)**2 + (y2 - trunk_center_x)**2 > (trunk_radius + 0.5)**2:
                        leaf_set.add((x2, y2, z2))
                    # favor downward movement for droop
                    if rng.random() < 0.45:
                        z2 += 1
                    else:
                        z2 += -1 if rng.random() < 0.4 else 0
                    # small lateral jitter
                    x2 += rng.choice([-1, 0, 1])
                    y2 += rng.choice([-1, 0, 1])
        # Write leaf voxels into the voxel grid avoiding trunk voxels
        if leaf_set:
            leaf_idxs = leaf_indices if leaf_indices else [9]
            for (lx, ly, lz) in leaf_set:
                if 0 <= lx < grid_size and 0 <= ly < grid_size and 0 <= lz < grid_size:
                    # avoid overwriting trunk voxels
                    if (lx, ly, lz) in trunk_voxels:
                        continue
                    voxels[lx, ly, lz] = leaf_idxs[rng.randint(0, len(leaf_idxs)-1)]

    # Instead of a layered solid trunk, use recursive branch drawing to build a more organic trunk
    # This mirrors the trunk construction used in the Treegen core: branches() draws the central
    # trunk as the main axis (starting with direction (0,0,1)) and records trunk voxels via draw_line.
    cx = grid_size // 2
    # start recursion which will populate trunk_voxels
    branches(cx, cx, 0, 0, 0, 1, 1)

    # After branches, estimate trunk_height and trunk_radius from generated trunk voxels
    if trunk_voxels:
        zs = [z for (_, _, z) in trunk_voxels]
        trunk_height = max(zs)
        # estimate trunk radius at base by checking voxels at low z levels
        base_voxels = [ (x,y) for (x,y,z) in trunk_voxels if z <= min(2, trunk_height) ]
        if base_voxels:
            cx0 = grid_size // 2
            max_r = 0
            for x,y in base_voxels:
                r = math.hypot(x - cx0, y - cx0)
                if r > max_r: max_r = r
            trunk_radius = max(1, int(math.ceil(max_r)))
        else:
            trunk_radius = 1
    else:
        trunk_height = min(grid_size - 2, max(8, int(size * iters * 0.9)))
        trunk_radius = max(1, int(math.ceil(params.get('trunksize', 1.0) * params.get('size', 1.0) * 2)))

    # Now add leaves using the recorded branch tips
    add_leaves()

    # Identify trunk surface voxels for potential bark scars
    trunk_voxels = set(trunk_voxels)
    surface_voxels = []
    for (x, y, z) in trunk_voxels:
        # a voxel is on surface if any 6-neighbor is not part of trunk
        is_surface = False
        for nx, ny, nz in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if (nx, ny, nz) not in trunk_voxels:
                is_surface = True
                break
        if is_surface:
            surface_voxels.append((x, y, z))

    # Choose a small number of random surface voxels to be bark scars
    scar_count = max(1, int(len(surface_voxels) * 0.02))  # ~2% of surface voxels
    scars = set(rng.sample(surface_voxels, min(scar_count, len(surface_voxels)))) if surface_voxels else set()

    # Write trunk voxels; use scar palette index for scar positions when available
    trunk_list = list(trunk_voxels)
    rng.shuffle(trunk_list)
    # choose scar index if available (prefer last trunk index as scar color)
    scar_idx = trunk_indices[-1] if trunk_indices else 1
    for i, (x, y, z) in enumerate(trunk_list):
        if (x, y, z) in scars:
            voxels[x, y, z] = scar_idx
        else:
            idx = trunk_indices[i % len(trunk_indices)] if trunk_indices else 1
            voxels[x, y, z] = idx

    return voxels, palette


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


def generate_birchgen_preview(params, palette_name, grid_size=PREVIEW_GRID, view='front', progress_callback=None, cancel_check=None):
    """Produce a preview image that matches exported output by generating
    at full resolution then downscaling. Falls back to reduced-grid preview on error."""
    try:
        vox, palette = generate_birchgen_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=progress_callback, cancel_check=cancel_check)
        img_full = project_voxels_to_image(vox, palette, GRID, view=view)
        return img_full.resize((grid_size*3, grid_size*3), Image.NEAREST)
    except CancelledError:
        raise
    except Exception:
        shrink = grid_size / GRID
        params_preview = params.copy()
        params_preview['size'] *= shrink
        params_preview['trunksize'] *= shrink
        vox, palette = generate_birchgen_tree(params_preview, palette_name, grid_size=grid_size, preview=True, progress_callback=progress_callback, cancel_check=cancel_check)
        img = project_voxels_to_image(vox, palette, grid_size, view=view)
        return img.resize((grid_size*3, grid_size*3), Image.NEAREST)


def orient_voxels_for_export(voxels, view='front'):
    """Return reoriented copy of voxels so MagicaVoxel front matches preview.
    'front' -> identity, 'top' -> swap x<->z.
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


def export_birch(params, palette_name, prefix='birch', export_view='front'):
    voxels, palette = generate_birchgen_tree(params, palette_name, grid_size=GRID, preview=True)
    voxels_oriented = orient_voxels_for_export(voxels, view=export_view)
    exporter = VoxExporter(params)
    loaded_palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name) if palette_name else (palette, [9,17], [57,65])
    return exporter.export(voxels_oriented, loaded_palette, leaf_indices, trunk_indices, prefix, preview=False)


# Birch preset and CLI
def birch_preset(seed=None):
    if seed is None:
        seed = random.randint(1, 9999)
    params = {
        'seed': int(seed),
        # slightly shorter overall scale for a medium-height birch
        'size': 0.8,  # Reduced from 1.1 to 0.8 to lower the maximum scale
        # slender trunk
        'trunksize': 0.35,
        # moderately open crown
        'spread': 0.4,  # adjusted default for balance
        # low branch twisting
        'twisted': 0.1,
        # modest leaf density (paper birch has many small leaves but not overly dense)
        'leaves': 1.0,
        # slight droop to branches/foliage
        'gravity': 0.25,
        # moderate branching complexity
        'iterations': 10,
        # bias toward shorter outer branches
        'wide': 0.3,  # reduced default for less extreme wideness
    }
    return params


def save_preview_image(img, out_dir='output', tag='birch'):
    os.makedirs(out_dir, exist_ok=True)
    fn = f"{tag}_preview_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.png"
    path = os.path.join(out_dir, fn)
    img.save(path)
    return path