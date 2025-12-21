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

class VoxExporter:
    def __init__(self, params, palette_map=None, palette_subdir='tree', output_subdir='birch'):
        # Ensure both `palette_subdir` and `output_subdir` are stored; default to 'tree'
        self.params = params
        self.palette_map = palette_map or {'default': {'leaves':[9,17],'trunk':[57,65]}}
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
            pass
        # Fallback: return a simple grayscale palette and mapping from palette_map
        palette = [(i, i, i, 255) for i in range(256)]
        config = self.palette_map.get(key, next(iter(self.palette_map.values()))) if self.palette_map else {'leaves': [9,17], 'trunk': [57,65]}
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

def generate_birchgen_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=None, cancel_check=None):
    # Use local RNG so we don't reseed the global generator
    seed = int(params.get('seed', 1))
    rng = random.Random(seed)

    exporter = VoxExporter(params)
    palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name) if palette_name else ([(i,i,i,255) for i in range(256)], [9,17], [57,65])

    voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    gLeaves = []

    iters = max(int(params.get('iterations', 1)), 10)
    size = 150 * params.get('size', 1.0) / iters
    # Read trunk height parameter and calculate total height
    trunk_height_param = clamp(params.get('trunkheight', 1.0), 0.0, 5.0)
    # Branch density control (exposed by the UI as 'branchdensity')
    branch_density = clamp(params.get('branchdensity', 1.0), 0.0, 3.0)
    # Branch length control (exposed by the UI as 'branchlength')
    branch_length_param = clamp(params.get('branchlength', 1.0), 0.0, 3.0)
    # Branch direction control (exposed by the UI as 'branchdir') - tilts branches on x-axis for droop
    branch_dir_param = clamp(params.get('branchdir', -0.5), -5.0, 5.0)
    # Twisted controls whole-tree twist (rotate branches around vertical axis)
    twist_param = clamp(params.get('twisted', 0.5), 0.0, 4.0)
    # Leaf radius control (exposed by the UI as 'leaf_radius')
    leaf_radius_param = clamp(params.get('leaf_radius', 2.0), 1.0, 4.0)
    # Leaf stretch control (exposed by the UI as 'leaf_stretch') - controls vertical elongation of leaves
    leaf_stretch_param = clamp(params.get('leaf_stretch', 1.5), 0.5, 3.0)
    # Leaf gravity control (exposed by the UI as 'leaf_gravity') - controls directional gravity of leaves (up/down)
    leaf_gravity_param = clamp(params.get('leaf_gravity', 0.0), -1.0, 1.0)
    # Use a smaller trunk thickness multiplier for birch to keep trunks slender.
    # Scale by a reduced multiplier (3) rather than the Treegen core value to avoid overly thick trunks.
    gTrunkSize = (params.get('trunksize', 1.0) * 0.7 + 0.3) * params.get('size', 1.0) * 3
    spread = clamp(params.get('spread', 0.4), 0.3, 0.8)  # moderate spread
    bend = clamp(params.get('bend', 0.0), 0.0, 1.0)
    # Recompute branch length bases using `spread` (removed `wide` parameter)
    gBranchLength0 = size * (1 - spread * 0.5)
    gBranchLength1 = size * (spread * 0.5 + spread * 0.1)  # simplified influence
    # force a trunk-first phase (no branching) for birch
    TRUNK_PHASE = int(iters * 0.35)

    def normalize(x, y, z):
        l = math.sqrt(x*x + y*y + z*z)
        return (x/l, y/l, z/l) if l > 0 else (0,0,1)

    def rotate_y(x, z, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return x * c - z * s, x * s + z * c

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
        # shorten outer branches for birch (whippy, short outer branches)
        base_length = gBranchLength0 * (1 - t * 0.5) + gBranchLength1 * t
        
        if i <= TRUNK_PHASE:
            # Trunk segments: scale by trunk height parameter only
            height_scale = 0.5 + trunk_height_param * 0.3  # maps 0.0->0.5, 1.0->0.8, 5.0->2.0
            return base_length * height_scale
        else:
            # Branch segments: scale by branch length parameter only
            return base_length * branch_length_param

    def get_branch_size(i):
        t = math.sqrt((i - 1) / iters)
        return (1 - t) * gTrunkSize

    def get_branch_angle(i):
        # angle ramps up only after trunk phase; bias angles upward (in radians range)
        t = (i - TRUNK_PHASE) / max(1, (iters - TRUNK_PHASE))
        return clamp(0.15 + t * 0.25, 0.15, 0.4)

    def get_branch_prob(i):
        if i <= TRUNK_PHASE:
            return 0.0
        t = (i - TRUNK_PHASE) / max(1, iters - TRUNK_PHASE)
        base = clamp(0.3 + t * 0.5, 0.0, 1.0)
        # scale branching probability to control density, clamp to [0,1]
        return clamp(base * branch_density, 0.0, 1.0)

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
            # enforce trunk-only phase: grow straight up without branching
            if i <= TRUNK_PHASE:
                branches(x1, y1, z1, dx, dy, dz, i + 1)
                # update progress and return early
                progress_done += 1
                if progress_callback:
                    try:
                        progress_callback(min(progress_done / progress_total, 1.0))
                    except Exception:
                        pass
                return

            # determine lateral variance (always define var before use)
            # lateral variance driven by spread only; wide removed and twisted no longer affects variance
            var = i * 0.12 * (spread * 0.9 if 'spread' in locals() else 0.0)
            # let bend amplify lateral variance
            var *= (1.0 + bend * 0.6)

            # decide how many child branches; bias occasionally to 2 and possibly use a larger angle
            b = 1 + (rng.random() < 0.4)
            if rng.random() < get_branch_prob(i) * 0.6:
                b = max(b, 2)
                # use angle-derived variance for occasional larger-angle splits
                var = max(var, get_branch_angle(i))

            for _ in range(b):
                # Apply whole-tree twist by rotating the parent direction around Y before spawning child directions.
                # Angle scales with iteration so higher branches twist more; twist_param controls overall strength.
                angle = twist_param * 0.15 * (i / max(1, iters))
                rot_dx, rot_dz = rotate_y(dx, dz, angle)
                base_dx, base_dy, base_dz = rot_dx, dy, rot_dz
                # bias child directions upward and allow bend to influence lateral components
                dx2 = base_dx + rng.uniform(-var, var) + bend * 0.25 * rng.uniform(-1.0, 1.0) + branch_dir_param * 0.05
                dy2 = base_dy + rng.uniform(-var, var) + bend * 0.25 * rng.uniform(-1.0, 1.0)
                dz2 = base_dz + rng.uniform(-var, var * 0.6) - bend * 0.05 * rng.random()
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
                # start near the leaf source and perform a random walk to populate a cluster
                # scale initial offset by leaf radius
                radius_int = int(leaf_radius_param)
                # apply leaf stretch to z-direction for vertical elongation
                stretch_radius_z = int(radius_int * leaf_stretch_param)
                x2 = x1 + rng.randint(-radius_int, radius_int)
                y2 = y1 + rng.randint(-radius_int, radius_int)
                z2 = z1 + rng.randint(-stretch_radius_z, stretch_radius_z)
                steps = max(2, int(5 * params.get('leaves', 1.0)))
                for _ in range(steps):
                    # avoid placing leaves inside the trunk radius
                    if (x2 - trunk_center_x)**2 + (y2 - trunk_center_x)**2 > (trunk_radius + 0.5)**2:
                        leaf_set.add((x2, y2, z2))
                    # apply leaf gravity for directional preference (up/down)
                    gravity_prob = 0.5 + leaf_gravity_param * 0.5  # maps -1.0->0.0, 0.0->0.5, 1.0->1.0
                    if rng.random() < gravity_prob:
                        z2 -= 1  # upward movement
                    else:
                        z2 += 1  # downward movement
                    # scale lateral jitter by leaf radius, and z-jitter by stretch
                    x2 += rng.choice([-radius_int, 0, radius_int])
                    y2 += rng.choice([-radius_int, 0, radius_int])
                    z2 += rng.choice([-stretch_radius_z, 0, stretch_radius_z])

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
        trunk_height = min(grid_size - 2, max(8, int(size * iters * 0.9 * (0.5 + trunk_height_param * 0.3))))
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
    """
    Birch worker preview wrapper. Returns PIL.Image when possible.
    """
    try:
        vox, palette = generate_birchgen_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=progress_callback, cancel_check=cancel_check)
        img_full = project_voxels_to_image(vox, palette, GRID, view=view)
        return img_full.resize((grid_size * 3, grid_size * 3), Image.NEAREST)
    except CancelledError:
        raise
    except Exception:
        shrink = grid_size / GRID
        params_preview = params.copy()
        params_preview['size'] *= shrink
        params_preview['trunksize'] *= shrink
        vox, palette = generate_birchgen_tree(params_preview, palette_name, grid_size=grid_size, preview=True, progress_callback=progress_callback, cancel_check=cancel_check)
        img = project_voxels_to_image(vox, palette, grid_size, view=view)
        return img.resize((grid_size * 3, grid_size * 3), Image.NEAREST)


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