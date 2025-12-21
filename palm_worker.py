import os
import math
import random
import struct
from datetime import datetime

from PIL import Image
import numpy as np

try:
    from palette_worker import get_internal_palette, _palette_manager
except Exception:
    get_internal_palette = None
    _palette_manager = None

# Constants
GRID = 256
PREVIEW_GRID = 64

# resource helper
import sys as _sys

def resource_path(filename):
    if hasattr(_sys, '_MEIPASS'):
        return os.path.join(_sys._MEIPASS, filename)
    return filename

# utility
def clamp(v, a, b):
    return max(a, min(b, v))

# VoxExporter similar to other workers
class VoxExporter:
    def __init__(self, params, palette_map=None, palette_subdir='palm', output_subdir='palm'):
        # Ensure both `palette_subdir` and `output_subdir` are stored; defaults to 'palm'
        self.params = params
        self.palette_map = palette_map or {'default': {'leaves':[9,17],'trunk':[57,65]}}
        self.palette_subdir = palette_subdir
        self.output_subdir = output_subdir

    def load_palette(self, palette_name):
        # Prefer internal palette registry; do not load palettes from external files.
        key = os.path.basename(palette_name) if palette_name else 'default'
        try:
            if get_internal_palette and _palette_manager and key in _palette_manager.list_palettes():
                palette, mapping = get_internal_palette(key)
                return palette, mapping.get('leaves', [9,17]), mapping.get('trunk', [57,65])
        except Exception:
            pass
        # Fallback: simple grayscale palette and default indices
        palette = [(i, i, i, 255) for i in range(256)]
        return palette, [9,17], [57,65]

    def export(self, voxels, palette, leaf_indices, trunk_indices, prefix='palm', preview=False):
        if preview:
            return voxels, palette
        coords = np.argwhere(voxels > 0)
        voxel_data = bytearray()
        if coords.size == 0:
            dims = np.array([1,1,1], dtype=int)
            count = 0
        else:
            min_xyz = coords.min(axis=0).astype(int)
            max_xyz = coords.max(axis=0).astype(int)
            dims = (max_xyz - min_xyz + 1).astype(int)
            rel_coords = []
            for x,y,z in coords:
                c = int(voxels[x,y,z])
                x0 = int(x - min_xyz[0])
                y0 = int(y - min_xyz[1])
                z0 = int(z - min_xyz[2])
                rel_coords.append((x0,y0,z0,c))
            rel_coords.sort(key=lambda t: (t[2], t[1], t[0]))
            for x0,y0,z0,c in rel_coords:
                voxel_data += struct.pack('<4B', x0, y0, z0, c)
            count = len(rel_coords)
        # palette shift for MagicaVoxel
        if len(palette) >= 256:
            palette = palette[1:256] + [(0,0,0,0)]
        else:
            palette = palette[1:] + [(0,0,0,0)]
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
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join('output', self.output_subdir)
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f'{prefix}_{timestamp}.vox')
        with open(filename, 'wb') as f:
            f.write(vox_file)
        return filename

# ---------------- Palm generation ----------------

# Utility vector funcs
def normalize(v):
    x,y,z = v
    l = math.sqrt(x*x+y*y+z*z)
    if l < 1e-9:
        return (0.0,1.0,0.0)
    return (x/l, y/l, z/l)

def cross(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

def dot(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return ax * bx + ay * by + az * bz

# Small math helpers
def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))

def sstep(a, b, x):
    if b == a:
        return 1.0 if x >= b else 0.0
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)

def hash01(a, b, c, d=0.0):
    s = math.sin(a * 12.9898 + b * 78.233 + c * 37.719 + d * 11.13) * 43758.5453
    return s - math.floor(s)

def gravityTerms(t, gGain, gPow):
    ramp = sstep(0.0, 0.60, t)
    k2 = gGain * (t ** gPow) * ramp
    return k2, k2 * 0.65

def outwardBase(outScale):
    return 0.04 * outScale

def outwardExtra(t, isLow):
    return (0.06 * (1.0 - sstep(0.0, 0.30, t))) if isLow else 0.0

def frameUpdate(tdx, tdy, tdz, rx, ry, rz, wx, wy, wz):
    dd = tdx * rx + tdy * ry + tdz * rz
    prx, pry, prz = rx - tdx * dd, ry - tdy * dd, rz - tdz * dd
    prl = math.sqrt(prx * prx + pry * pry + prz * prz)
    if prl < 1e-6:
        prx, pry, prz = cross((tdx, tdy, tdz), (wx, wy, wz))
        prl = math.sqrt(prx * prx + pry * pry + prz * prz)
    rx, ry, rz = prx / prl, pry / prl, prz / prl
    bx, by, bz = cross((tdx, tdy, tdz), (rx, ry, rz))
    return rx, ry, rz, bx, by, bz

def leafletLengthFactor(t01, az, phase, L):
    hz = 0.8 + 0.6 * (L / 120.0)
    m = 0.5 + 0.5 * math.cos(2 * math.pi * (hz * t01 + phase))
    ease = 0.5 - 0.5 * math.cos(math.pi * m)
    R = (1.0 - 1.0)  # pHealth not available here; will be passed in when used
    r_local = R * ease
    keep = 1.0 - r_local
    return max(0.04, keep)

def curvesForH(h):
    t = sstep(0.0, 1.0, h ** (0.5 + 1.5 * (1.0 - 0.65)))
    phiMinDeg = lerp(1.0, 35.0, t)
    phiMaxDeg = lerp(12.0, 86.0, t)
    rStart = lerp(0.18, 1.0, t)
    outScale = lerp(0.30, 1.0, t)
    sagDelay = lerp(0.50, 0.20, t)
    safeDelay = lerp(0.50, 0.25, t)
    avoidScale = lerp(0.75, 1.00, t)
    return phiMinDeg, phiMaxDeg, rStart, outScale, sagDelay, safeDelay, avoidScale

def samplePhiForH(h, k, az):
    u = hash01(0.73 * 1.0, k * 1.11, az * 0.37, 9.0)
    phiMinDeg, phiMaxDeg, _, _, _, _, _ = curvesForH(h)
    span = max(0.0, phiMaxDeg - phiMinDeg)
    e = lerp(0.45, 1.60, sstep(0.0, 1.0, h))
    uT = u ** e
    deg = phiMinDeg + span * uT
    return math.radians(deg)

class CancelledError(Exception):
    pass

def generate_palm_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=None, cancel_check=None):
    # local RNG
    seed = int(params.get('seed', 1))
    rng = random.Random(seed)

    exporter = VoxExporter(params)
    palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name) if palette_name else ([(i,i,i,255) for i in range(256)], [9,17], [57,65])

    voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

    # read params with defaults and clamp
    pSize = clamp(params.get('size', 1.0), 0.1, 3.0)
    pTrunkExtend = clamp(params.get('trunkextend', 80.0), 0.0, 340.0)
    pTrunkSize = clamp(params.get('trunksize', 1.0), 0.3, 2.0)
    pTrunkIter = int(clamp(params.get('trunkiter', 40), 12, 80))
    pBend = clamp(params.get('bend', 1.0), 0.0, 1.0)
    pLeafCount = int(clamp(params.get('leafcount', 33), 4, 72))
    pLeafLengthScale = clamp(params.get('leaflength', 0.35), 0.1, 3.0)
    pLeafVar = clamp(params.get('leafvar', 0.50), 0.0, 1.0)
    pGravity = clamp(params.get('gravity', 0.30), 0.0, 1.0)
    pLeafWidth = clamp(params.get('leafwidth', 0.25), 0.1, 1.0)
    pLeafFill = clamp(params.get('leaffill', 0.40), 0.0, 1.0)
    pHealth = clamp(params.get('health', 1.0), 0.0, 1.0)
    pLeafWobble = clamp(params.get('leafwobble', 0.00), 0.0, 1.0)
    pLeafletDip = clamp(params.get('leafletdip', 1.00), 0.0, 1.0)
    pFrondRandom = clamp(params.get('frondrandom', 1.0), 0.0, 1.0)
    # Scale factor to amplify effect of the frond randomness slider (keeps slider 0..1)
    pFrondFactor = 1.0 + 2.0 * pFrondRandom  # at max, effects up to ~3x stronger

    # Derived
    MIN_H = 60.0
    H = MIN_H + pTrunkExtend
    segL = H / pTrunkIter
    baseR = 6.0 * pTrunkSize * pSize

    CX, CY, CZ, CR = [0]*(pTrunkIter+2), [0]*(pTrunkIter+2), [0]*(pTrunkIter+2), [0]*(pTrunkIter+2)

    trunk_voxels = set()
    # Map of leaf voxel coordinate -> best fractional position along frond (t01 0..1).
    # We store the t01 value that is closest to the frond center (0.5) so overlapping
    # strokes favor central leaflet contribution for colouring.
    leaf_voxels = {}
     
    def line_trunk(x0,y0,z0, x1,y1,z1, r0, r1):
        steps = max(1, int(math.dist((x0,y0,z0),(x1,y1,z1))*2))
        for i in range(steps+1):
            t = i/steps
            x = x0 + (x1-x0)*t; y = y0 + (y1-y0)*t; z = z0 + (z1-z0)*t
            r = r0 + (r1-r0)*t
            ir = int(math.ceil(r))
            for dx in range(-ir, ir+1):
                for dy in range(-ir, ir+1):
                    for dz in range(-ir, ir+1):
                        if dx*dx+dy*dy+dz*dz <= r*r:
                            xi = int(round(x+dx)); yi = int(round(y+dy)); zi = int(round(z+dz))
                            if 0 <= xi < grid_size and 0 <= yi < grid_size and 0 <= zi < grid_size:
                                trunk_voxels.add((xi, yi, zi))
 
    def line_leaf(x0,y0,z0, x1,y1,z1, r0, r1, t_frac=None):
        steps = max(1, int(math.dist((x0,y0,z0),(x1,y1,z1))*2))
        for i in range(steps+1):
            t = i/steps
            x = x0 + (x1-x0)*t; y = y0 + (y1-y0)*t; z = z0 + (z1-z0)*t
            r = r0 + (r1-r0)*t
            ir = int(math.ceil(r))
            for dx in range(-ir, ir+1):
                for dy in range(-ir, ir+1):
                    for dz in range(-ir, ir+1):
                        if dx*dx+dy*dy+dz*dz <= r*r:
                            xi = int(round(x+dx)); yi = int(round(y+dy)); zi = int(round(z+dz))
                            if 0 <= xi < grid_size and 0 <= yi < grid_size and 0 <= zi < grid_size:
                                # store the fractional position (t_frac) for later color mapping.
                                # Prefer the stored t that is closer to the frond center (0.5).
                                if t_frac is None:
                                    v = 0.5
                                else:
                                    v = float(t_frac)
                                prev = leaf_voxels.get((xi, yi, zi))
                                if prev is None or abs(v - 0.5) < abs(prev - 0.5):
                                    leaf_voxels[(xi, yi, zi)] = v

    # build trunk
    x,y,z = grid_size//2, 0, grid_size//2
    dx,dy,dz = 0.0, 1.0, 0.0
    leanAz = rng.random() * 2*math.pi
    lx, lz = math.cos(leanAz), math.sin(leanAz)
    bendVar = 0.10 * pBend
    leanBase = 0.008 + 0.022*pBend
    leanPow = 2.2 + 1.0*pBend

    def radiusAt01(t01):
        NAT_TAPER = 0.14
        return baseR * (1.0 - NAT_TAPER * (t01 ** 1.15))

    for i in range(1, pTrunkIter+1):
        t = i / pTrunkIter
        lean = leanBase * (t ** leanPow)
        dx = dx + rng.uniform(-bendVar, bendVar) + lean*lx
        dz = dz + rng.uniform(-bendVar, bendVar) + lean*lz
        nxn = math.sqrt(dx*dx + dy*dy + dz*dz)
        dx,dy,dz = dx/nxn, dy/nxn, dz/nxn
        x1 = x + dx*segL; y1 = y + dy*segL; z1 = z + dz*segL
        t0 = (i-1)/pTrunkIter; t1 = i/pTrunkIter
        r0 = radiusAt01(t0); r1 = radiusAt01(t1)
        line_trunk(x,y,z, x1,y1,z1, r0, r1)
        CX[i], CY[i], CZ[i], CR[i] = x1, y1, z1, r1
        x,y,z = x1,y1,z1
    CX[pTrunkIter+1], CY[pTrunkIter+1], CZ[pTrunkIter+1] = CX[pTrunkIter], CY[pTrunkIter], CZ[pTrunkIter]
    CR[pTrunkIter+1] = CR[pTrunkIter]

    # simple canopy: place frond axes and draw fronds as lines with leaf voxels
    cAnchorX, cAnchorY, cAnchorZ = CX[pTrunkIter], CY[pTrunkIter], CZ[pTrunkIter]
    # basis from axis
    def makeBasisFromAxis(wx,wy,wz):
        w = normalize((wx,wy,wz))
        ax = (0.0,1.0,0.0)
        cx = cross(w, ax)
        clen = math.sqrt(cx[0]*cx[0] + cx[1]*cx[1] + cx[2]*cx[2])
        if clen < 1e-5:
            ax = (1.0,0.0,0.0)
            cx = cross(w, ax)
            clen = math.sqrt(cx[0]*cx[0] + cx[1]*cx[1] + cx[2]*cx[2])
        ux = (cx[0]/clen, cx[1]/clen, cx[2]/clen)
        vx = cross(w, ux)
        vx = normalize(vx)
        return ux, vx, w

    ux, vx, w = makeBasisFromAxis(dx,dy,dz)
    wx,wy,wz = w

    # crown radius based on trunk top radius
    crownRad = max(0.2, (CR[pTrunkIter] if CR[pTrunkIter] else baseR) * 0.15)

    baseLeafL = 120.0 * pSize * pLeafLengthScale

    # Full frond generator
    def generate_frond(azimuth, phi, L, leaf_index, rStart, outScale, sagDelay, safeDelay, avoidScale, lowSagScale):
        # local rng usage
        # build local basis R from ux/vx and given azimuth
        cosA = math.cos(azimuth); sinA = math.sin(azimuth)
        Rx = (ux[0] * cosA + vx[0] * sinA, ux[1] * cosA + vx[1] * sinA, ux[2] * cosA + vx[2] * sinA)
        # start point slightly offset from crown
        sx = cAnchorX + Rx[0] * (crownRad * rStart * 0.5)
        sy = cAnchorY + Rx[1] * (crownRad * rStart * 0.5)
        sz = cAnchorZ + Rx[2] * (crownRad * rStart * 0.5)

        cosP = math.cos(phi); sinP = math.sin(phi)
        dx_f, dy_f, dz_f = normalize((wx * cosP + Rx[0] * sinP, wy * cosP + Rx[1] * sinP, wz * cosP + Rx[2] * sinP))

        # initial right/binormal
        rx, ry, rz = cross((dx_f, dy_f, dz_f), (wx, wy, wz))
        rl = math.sqrt(rx * rx + ry * ry + rz * rz)
        if rl < 1e-6:
            rx, ry, rz = 1.0, 0.0, 0.0
            rl = 1.0
        rx, ry, rz = rx / rl, ry / rl, rz / rl
        bx, by, bz = cross((dx_f, dy_f, dz_f), (rx, ry, rz))

        steps = 48
        seg = L / steps

        # stem thickness heuristics
        stemBase = max(0.35, 0.22 * baseR)

        # leaflet spacing
        minStride = 0.36
        density = max(0.001, pLeafFill)
        spacing = minStride / density

        # Apply leaf variance parameter and frond-randomness: vary spacing, width per-frond
        leafVarLocal = max(0.0, min(1.0, pLeafVar))
        # amplify per-frond variation using pFrondFactor (moderate increases)
        spacing *= (1.0 + rng.uniform(-0.40, 0.40) * leafVarLocal * pFrondFactor)

        wMax = (0.35 * L) * pLeafWidth * (1.0 + rng.uniform(-0.40, 0.40) * leafVarLocal * pFrondFactor)
        leafletR0 = max(0.30, 0.10 * stemBase * (1.0 + rng.uniform(-0.25, 0.25) * leafVarLocal * pFrondFactor))

        sink = min(1.0, (CR[pTrunkIter] or baseR) * 0.20)
        px, py, pz = sx - wx * sink, sy - wy * sink, sz - wz * sink
        prevx, prevy, prevz = px, py, pz

        accLen = 0.0
        nextDrop = spacing
        dropIdx = 0

        # per-frond jitter/phase (scaled by frond randomness factor)
        bandPhase = rng.random() * 2 * math.pi
        sideJit = {1: rng.uniform(-0.25, 0.25) * pFrondFactor, -1: rng.uniform(-0.25, 0.25) * pFrondFactor}
        T = max(0.0, min(1.0, ((1.0 - pHealth) - 0.35) / 0.65))
        clusterRun = {1: 0, -1: 0}
        hPhase = rng.random() * 2 * math.pi

        rampVar = 0.90 + 0.20 * hash01(seed * 2.37, (leaf_index or 0) * 0.53, azimuth * 0.19, 8.0)
        rampSteps = max(2, int(math.floor(0.12 * rampVar * steps)))

        for i in range(1, steps + 1):
            t = i / steps
            # gravity and outward offsets
            k2_base, k2_curve = gravityTerms(t, (0.55 + 0.65 * max(0.0, math.sin(phi))) * pGravity, 3.0 * max(0.0, math.cos(phi)) + 0.6 * max(0.0, math.sin(phi)))
            gx, gy, gz = 0.0, -k2_curve, 0.0
            extraOut = outwardExtra(t, lowSagScale < 0.999)
            ox, oy, oz = Rx[0] * (outwardBase(outScale) + extraOut), Rx[1] * (outwardBase(outScale) + extraOut), Rx[2] * (outwardBase(outScale) + extraOut)
            tdx, tdy, tdz = normalize((dx_f + gx + ox, dy_f + gy + oy, dz_f + gz + oz))
            # clamp turn for low fronds (simple)
            if lowSagScale < 0.999:
                maxTurn = lerp(math.radians(22.0), math.radians(10.0), (t ** 0.6))
                pvx, pvy, pvz = normalize((dx_f, dy_f, dz_f))
                # simple slerp clamp if needed
                # compute angle
                c = max(-1.0, min(1.0, pvx * tdx + pvy * tdy + pvz * tdz))
                ang = math.acos(c)
                if ang > maxTurn:
                    s = maxTurn / (ang + 1e-9)
                    tdx, tdy, tdz = normalize((pvx * (1.0 - s) + tdx * s, pvy * (1.0 - s) + tdy * s, pvz * (1.0 - s) + tdz * s))

            if t < safeDelay:
                nx, ny, nz = px + tdx * seg, py + tdy * seg, pz + tdz * seg
            else:
                vx0 = (px + tdx * seg) - cAnchorX
                vy0 = (py + tdy * seg) - cAnchorY
                vz0 = (pz + tdz * seg) - cAnchorZ
                proj = vx0 * wx + vy0 * wy + vz0 * wz
                rx0, ry0, rz0 = vx0 - wx * proj, vy0 - wy * proj, vz0 - wz * proj
                rlen = math.sqrt(rx0 * rx0 + ry0 * ry0 + rz0 * rz0)
                avoidR = max((CR[pTrunkIter] or baseR) * 0.6 * avoidScale, 1.5 * avoidScale)
                if rlen < avoidR:
                    push = (avoidR - rlen) + 0.2
                    if rlen < 1e-6:
                        pxdir, pydir, pzdir = Rx[0], Rx[1], Rx[2]
                    else:
                        pxdir, pydir, pzdir = rx0 / rlen, ry0 / rlen, rz0 / rlen
                    nx, ny, nz = px + tdx * seg + pxdir * push, py + tdy * seg + pydir * push, pz + tdz * seg + pzdir * push
                    tdx, tdy, tdz = normalize((tdx + pxdir * 0.12, tdy + pydir * 0.12, tdz + pzdir * 0.12))
                else:
                    nx, ny, nz = px + tdx * seg, py + tdy * seg, pz + tdz * seg

            # record wood segment (use line_trunk)
            line_trunk(prevx, prevy, prevz, nx, ny, nz, max(0.05, stemBase * (1.0 - (t ** 0.7))), max(0.2, stemBase * 0.12))

            # update frame
            rx, ry, rz, bx, by, bz = frameUpdate(tdx, tdy, tdz, rx, ry, rz, wx, wy, wz)

            ds = math.sqrt((nx - px) ** 2 + (ny - py) ** 2 + (nz - pz) ** 2)
            accLen += ds

            # place leaflets based on spacing after petiole
            petioleSteps = max(1, int(math.floor((0.12 + (0.08 if lowSagScale < 0.999 else 0.0)) * steps)))
            tipStopStep = max(petioleSteps + 1, steps - int(math.floor(0.5 * 0.05 * steps)))
            if i > petioleSteps and i < steps - 1 and i <= tipStopStep:
                while accLen >= nextDrop:
                    tLocal = (nextDrop - (accLen - ds)) / ds
                    t01 = (i - 1 + tLocal) / steps
                    wEnv = wMax * math.sin(math.pi * max(0.0, min(1.0, t01)))
                    wEnv = wEnv * (1.0 + 0.06 * (pLeafFill - 0.5))
                    grow = 1.0
                    stepIdxFloat = (i - 1 + tLocal)
                    afterPetiole = stepIdxFloat - petioleSteps
                    if afterPetiole > 0 and afterPetiole < rampSteps:
                        grow = max(0.0, min(1.0, afterPetiole / rampSteps))
                    keepBase = 1.0
                    j = 1.0 - 0.08 * (1.0 - pHealth) * ((hash01(seed, azimuth, dropIdx, 7.0) * 2.0) - 1.0)
                    keep = max(0.0, min(1.0, keepBase * j))
                    w = wEnv * keep * grow
                    # per-leaflet variation influenced by pLeafVar to affect size/jitter
                    if leafVarLocal > 0.0:
                        # increase per-leaflet jitter amplitude modestly and scale by frond factor
                        jitter = (hash01(seed, azimuth, dropIdx, 11.0) - 0.5) * 0.8 * leafVarLocal * pFrondFactor
                        w = max(0.0, w * (1.0 + jitter))

                    bothSides = (pLeafFill >= 0.95)

                    def draw_side(signSide):
                        nonlocal dropIdx
                        lr = leafletR0 * (1.0 - 0.85 * (t01 ** 1.1))
                        leafDroop = (0.15 + 0.55 * pGravity) * (t01 ** 0.9)
                        backPad = min(0.95 * spacing, 0.8 * seg)
                        cbx, cby, cbz = nx - tdx * backPad, ny - tdy * backPad, nz - tdz * backPad
                        railOff = (max(0.05, stemBase * (1.0 - (t ** 0.7))) * 0.55) + 0.35
                        sbx, sby, sbz = cbx + rx * railOff * signSide, cby + ry * railOff * signSide, cbz + rz * railOff * signSide
                        nibLen = max(0.28, 0.38 * spacing)
                        nbx, nby, nbz = sbx - tdx * nibLen, sby - tdy * nibLen, sbz - tdz * nibLen
                        nibR = max(0.40, lr * 0.9)
                        line_leaf(sbx, sby, sbz, nbx, nby, nbz, nibR, nibR, t_frac=t01)
                        und = 0.0
                        if pLeafWobble > 0.0:
                            und = (0.55 * pLeafWobble * sstep((0.65 - 0.30 * pGravity) * 0.6, (0.65 - 0.30 * pGravity), t01)) * math.sin(2 * math.pi * (1.2 + 1.6 * pLeafWobble) * t01 + hPhase)
                        und = und * (1.0 if signSide == 1 else -0.85) * w * pFrondFactor
                        ex = sbx + rx * w * signSide + bx * und
                        ey = sby + ry * w * signSide + by * und - w * leafDroop
                        ez = sbz + rz * w * signSide + bz * und
                        if pLeafletDip > 0.001:
                            enter = sstep(0.10, 0.25, t01)
                            tail = 1.0 - 0.70 * sstep(0.75, 1.00, t01)
                            along = enter * tail
                            gravK = 0.65 + 0.45 * pGravity
                            lowK = 1.20 if lowSagScale < 0.999 else 1.0
                            jit = 0.90 + 0.20 * (hash01(seed * 3.7, azimuth * 0.23, dropIdx * 0.11, signSide * 0.17) - 0.5)
                            dipStrength = pLeafletDip * along * gravK * lowK * jit
                            ey = ey - (dipStrength * w)
                        line_leaf(sbx, sby, sbz, ex, ey, ez, lr, max(0.2, lr * 0.8), t_frac=t01)
                        dropIdx += 1

                    if w > 0.0:
                        if bothSides:
                            draw_side(1); draw_side(-1)
                        else:
                            draw_side(1 if (dropIdx % 2 == 0) else -1)

                    nextDrop += spacing

            prevx, prevy, prevz = nx, ny, nz
            px, py, pz = nx, ny, nz
            dx_f, dy_f, dz_f = tdx, tdy, tdz

        return

    # Fronds: ensure canopy arrays exist and generate
    try:
        AZ
    except NameError:
        # build simple fallback canopy with added randomness for natural variation
        AZ = []
        PHI = []
        LEN = []
        RST = []
        OBS = []
        SDLY = []
        FDLY = []
        AVO = []
        LSG = []
        for k in range(pLeafCount):
            # Azimuth: even spacing with wider jitter scaled by pFrondFactor
            base_az = (k / pLeafCount) * 2 * math.pi
            az_jitter = rng.uniform(-0.25, 0.25) * pFrondFactor
            AZ.append(base_az + az_jitter)
            
            # Phi (elevation angle): base 45 degrees with larger variation scaled by pFrondFactor
            base_phi = math.radians(45.0)
            phi_jitter = rng.uniform(-math.radians(25.0), math.radians(25.0)) * pFrondFactor
            PHI.append(base_phi + phi_jitter)
            
            # Length: base length with slightly larger variation scaled by pFrondFactor
            base_len = max(24.0, baseLeafL)
            len_var = 1.0 + rng.uniform(-0.3, 0.3) * pFrondFactor
            LEN.append(base_len * len_var)
            
            # Other parameters with slight variation scaled by pFrondFactor
            RST.append(1.0 + rng.uniform(-0.12, 0.12) * pFrondFactor)
            OBS.append(1.0 + rng.uniform(-0.12, 0.12) * pFrondFactor)
            SDLY.append(0.5 + rng.uniform(-0.12, 0.12) * pFrondFactor)
            FDLY.append(0.5 + rng.uniform(-0.12, 0.12) * pFrondFactor)
            AVO.append(1.0 + rng.uniform(-0.12, 0.12) * pFrondFactor)
            LSG.append(1.0 + rng.uniform(-0.12, 0.12) * pFrondFactor)

    for k in range(len(AZ)):
        generate_frond(AZ[k], PHI[k], LEN[k], k+1, RST[k], OBS[k], SDLY[k], FDLY[k], AVO[k], LSG[k])

    # Assign colors to voxels - trunk banding along vertical (Y)
    # Keep original trunk_voxels as a set; build a sorted list for deterministic processing.
    trunk_list = sorted(list(trunk_voxels), key=lambda t: t[1]) if trunk_voxels else []
    if trunk_list and trunk_indices:
        ys = [y for (_x, y, _z) in trunk_list]
        min_y = min(ys)
        max_y = max(ys)
        span = max(1, max_y - min_y)
        # number of cycles along the trunk (ensure at least some repetition)
        cycles = max(4, len(trunk_indices))
        n_colors = len(trunk_indices)
        for (x, y, z) in trunk_list:
            # normalized height
            t = (y - min_y) / span
            cycle_pos = t * cycles
            cycle_idx = int(math.floor(cycle_pos))
            s = cycle_pos - cycle_idx
            # Alternate direction each cycle for banding ripple
            if cycle_idx % 2 == 1:
                s = 1.0 - s
            # continuous index into trunk_indices
            idxf = s * (n_colors - 1)
            # tiny deterministic jitter via RNG to soften band edges
            idxf += (rng.random() - 0.5) * 0.08
            idxf = max(0.0, min(n_colors - 1, idxf))
            chosen = int(round(idxf))
            voxels[x, y, z] = trunk_indices[chosen]
    else:
        # fallback: spread available trunk indices across trunk voxels
        trunk_list = list(trunk_voxels)
        rng.shuffle(trunk_list)
        for i, (x, y, z) in enumerate(trunk_list):
            idx = trunk_indices[i % len(trunk_indices)] if trunk_indices else 1
            voxels[x, y, z] = idx

    # Map leaf voxels to palette indices using the recorded t01 fraction.
    # Darkness is highest at frond center (t01 ~= 0.5) and lighter toward both ends.
    leaf_items = list(leaf_voxels.items())  # [((x,y,z), t01), ...]
    rng.shuffle(leaf_items)
    n_colors = len(leaf_indices) if leaf_indices else 0
    for (x, y, z), t in leaf_items:
        if voxels[x, y, z] != 0:
            continue
        t_local = 0.5 if t is None else float(t)
        darkness = 1.0 - abs(t_local - 0.5) * 2.0
        darkness = max(0.0, min(1.0, darkness))
        if n_colors:
            idxf = darkness * (n_colors - 1)
            idxf += (rng.random() - 0.5) * 0.08
            idxf = max(0.0, min(n_colors - 1, idxf))
            chosen = int(round(idxf))
            voxels[x, y, z] = leaf_indices[chosen]
        else:
            voxels[x, y, z] = 1

    return voxels, palette

def generate_palm_preview(params, palette_name, grid_size=PREVIEW_GRID, view='front', progress_callback=None, cancel_check=None):
    try:
        vox, palette = generate_palm_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=progress_callback, cancel_check=cancel_check)
        if isinstance(vox, np.ndarray):
            vox = np.swapaxes(vox, 1, 2)
        img_full = project_voxels_to_image(vox, palette, GRID, view=view)
        return img_full.resize((grid_size * 3, grid_size * 3), Image.NEAREST)
    except CancelledError:
        raise
    except Exception:
        shrink = grid_size / GRID
        params_preview = params.copy()
        params_preview['size'] *= shrink
        params_preview['trunksize'] *= shrink
        vox, palette = generate_palm_tree(params_preview, palette_name, grid_size=grid_size, preview=True, progress_callback=progress_callback, cancel_check=cancel_check)
        if isinstance(vox, np.ndarray):
            vox = np.swapaxes(vox, 1, 2)
        img = project_voxels_to_image(vox, palette, grid_size, view=view)
        return img.resize((grid_size * 3, grid_size * 3), Image.NEAREST)

def orient_voxels_for_export(voxels, view='front'):
    try:
        if view == 'front':
            return np.swapaxes(voxels, 1, 2).copy()
        elif view == 'top':
            return np.swapaxes(voxels, 0, 2).copy()
        else:
            return np.swapaxes(voxels, 1, 2).copy()
    except Exception:
        return voxels


def export_palm(params, palette_name, prefix='palm', export_view='front'):
    # Scale params to fit the tree within the 256 grid for export
    params_scaled = params.copy()
    scale_factor = 0.6  # Adjust to ensure max extent <= 256
    params_scaled['size'] = params.get('size', 1.0) * scale_factor
    params_scaled['trunkextend'] = params.get('trunkextend', 80.0) * scale_factor
    params_scaled['leaflength'] = params.get('leaflength', 0.35) * scale_factor
    params_scaled['trunksize'] = params.get('trunksize', 1.0) * scale_factor
    voxels, palette = generate_palm_tree(params_scaled, palette_name, grid_size=GRID, preview=True)
    # Orient voxels for MagicaVoxel: swap Y/Z
    voxels_oriented = orient_voxels_for_export(voxels, view=export_view)
    exporter = VoxExporter(params_scaled)
    loaded_palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name) if palette_name else (palette, [9,17], [57,65])
    return exporter.export(voxels_oriented, loaded_palette, leaf_indices, trunk_indices, prefix, preview=False)

# Helper projection
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
