import os
import math
import random
from datetime import datetime
import struct

from PIL import Image
try:
    import numpy as np
except Exception:
    raise SystemExit('numpy is required for kapok_worker')

# Constants (match other workers)
GRID = 256
PREVIEW_GRID = 64

# reuse VoxExporter from treegen_worker if available
try:
    from treegen_worker import VoxExporter, project_voxels_to_image
    from treegen_worker import resource_path as _resource_path
except Exception:
    # minimal fallback implementations
    def _resource_path(p):
        return p

    # prefer internal palette manager when available
    try:
        from palette_worker import get_internal_palette, _palette_manager
    except Exception:
        get_internal_palette = None
        _palette_manager = None

    class VoxExporter:
        def __init__(self, params, palette_map=None, palette_subdir='tree', output_subdir='kapok', *args, **kwargs):
            # Backwards-compatible fallback initializer. Accepts legacy extra args and ignores them.
            self.params = params
            self.palette_map = palette_map or {}
            self.palette_subdir = palette_subdir
            self.output_subdir = output_subdir

        def load_palette(self, name):
            # Always prefer the internal palette registry when present
            key = os.path.basename(name) if name else 'default'
            try:
                if get_internal_palette and _palette_manager and key in _palette_manager.list_palettes():
                    palette, mapping = get_internal_palette(key)
                    return palette, mapping.get('leaves', [9,17]), mapping.get('trunk', [57,65])
            except Exception:
                pass
            # Fallback: simple default palette and indices
            pal = [(i, i, i, 255) for i in range(256)]
            return pal, [9,17], [57,65]

        def export(self, voxels, palette, leaf_indices, trunk_indices, prefix='kapok', preview=False):
            if preview:
                return voxels, palette
            # write a minimal binary to file for compatibility
            out_dir = os.path.join('output', self.output_subdir)
            os.makedirs(out_dir, exist_ok=True)
            filename = os.path.join(out_dir, f'{prefix}_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.vox')
            with open(filename, 'wb') as f:
                f.write(b'VOX')
            return filename

# Helper clamp

def clamp(a, mi, ma):
    if a < mi:
        return mi
    elif a > ma:
        return ma
    else:
        return a

class CancelledError(Exception):
    pass

def generate_kapok_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=None, cancel_check=None):
    pRandom = int(params.get('randomseed', params.get('seed', 1)))
    pTrunkExtend = clamp(float(params.get('trunkextend', 0.0)), 0.0, 340.0)
    pTrunkSize = clamp(float(params.get('trunksize', 1.0)), 0.3, 4.0)
    pTrunkIter = clamp(int(params.get('trunkiter', 40)), 8, 40)
    pBend = clamp(float(params.get('bend', 0.5)), 0.0, 1.0)
    pRootTwist = clamp(float(params.get('roottwist', 0.3)), 0.0, 1.0)
    pRootProfile = clamp(float(params.get('rootprofile', 0.5)), 0.0, 1.0)
    pRootSpread = clamp(float(params.get('rootspread', 0.5)), 0.0, 1.0)
    pRootCount = clamp(int(params.get('rootcount', 5)), 0, 8)
    pRootVariance = clamp(float(params.get('rootvariance', 1.0)), 0.0, 1.0)

    pCanopyIter = clamp(int(params.get('canopyiter', 12)), 5, 15)
    pSize = clamp(float(params.get('size', 1.0)), 0.1, 3.0)
    pWide = clamp(float(params.get('wide', 0.6)), 0.0, 1.0)
    pSpread = clamp(float(params.get('spread', 0.5)), 0.0, 1.0)
    pCanopyTwist = clamp(float(params.get('canopytwist', 0.5)), 0.0, 1.0)
    pLeaves = clamp(float(params.get('leaves', 0.7)), 0.0, 3.0)
    pGravity = clamp(float(params.get('gravity', 0.0)), -1.0, 1.0)
    pCanopyThick = clamp(float(params.get('canopythick', 0.25)), 0.0, 1.0)
    pCanopyProfile = clamp(float(params.get('canopyprofile', 0.9)), 0.0, 1.0)
    pCanopyFlat = clamp(float(params.get('canopyflat', 0.75)), 0.0, 1.0)
    pCanopyStart = clamp(float(params.get('canopystart', 0.75)), 0.0, 1.0)

    MIN_H = 60.0
    H = MIN_H + pTrunkExtend
    segL = H / pTrunkIter
    baseR = 6.0 * pTrunkSize * pSize
    bendVar = 0.06 * pBend
    NAT_TAPER = 0.18
    blendZoneFrac = 0.40
    cJoinBlendSteps = 2
    cJoinHoldSteps = 5

    rootHeightBase = 0.62 + 0.36 * pRootProfile
    spreadPow = 2.2
    spreadS = (pRootSpread ** spreadPow)
    fanBase = (0.25 + 2.20 * pRootProfile) * baseR * 0.60
    fanCapBase = fanBase * spreadS

    wedgeHalfA_Base = (0.12 + 0.18 * pRootProfile)
    angStepsBase = max(14, math.floor(22 + 18 * pRootProfile))
    ridgeAmpK_Base = 0.35 + 0.25 * pRootProfile
    ridgeSharp = 1.6

    sinkStartBase = 0.35 + 0.35 * pRootProfile
    holdUnderEps = 1.0
    seamOverlapVox = 3.2
    radialPad = 1.0

    CX = [0.0] * (pTrunkIter + 3)
    CY = [0.0] * (pTrunkIter + 3)
    CZ = [0.0] * (pTrunkIter + 3)
    CR = [0.0] * (pTrunkIter + 3)

    cAnchorX = cAnchorY = cAnchorZ = 0.0
    cAnchorDX = cAnchorDY = cAnchorDZ = 0.0
    cBaseScale = 1.0
    cJoinRadius = 1.0

    def smooth5(x):
        if x <= 0.0: return 0.0
        if x >= 1.0: return 1.0
        return x*x*x*(x*(x*6 - 15) + 10)

    def normalize(x,y,z):
        l = math.sqrt(x*x + y*y + z*z)
        if l < 1e-9: return 0.0,1.0,0.0
        return x/l, y/l, z/l

    def radiusAt01(t01):
        rNat = baseR * (1.0 - NAT_TAPER * (t01 ** 1.2))
        t0 = 1.0 - blendZoneFrac
        if t01 > t0:
            u = smooth5((t01 - t0) / blendZoneFrac)
            rTopNat = baseR * (1.0 - NAT_TAPER)
            rNat = rNat * (1.0 - u) + rTopNat * u
        return rNat

    def sampleAtY(yf):
        f = (yf / H) * pTrunkIter
        i = max(1, min(pTrunkIter-1, math.floor(f)))
        a = f - i
        x = CX[i] + (CX[i+1] - CX[i]) * a
        z = CZ[i] + (CZ[i+1] - CZ[i]) * a
        r = CR[i] + (CR[i+1] - CR[i]) * a
        return x, z, r

    # grid and brushes
    voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    matWood = 57
    matLeaf = 9

    # local RNG
    rng = random.Random()
    if pRandom != 0:
        rng.seed(pRandom)

    def line(x,y,z, x1,y1,z1, r0, r1):
        # similar to draw line in treegen_worker but 3D filled capsules
        steps = int(math.dist([x,y,z],[x1,y1,z1]) * 2)
        if steps == 0: steps = 1
        for i in range(steps+1):
            if cancel_check and cancel_check():
                raise CancelledError()
            t = i / steps
            cx = x + t*(x1-x)
            cy = y + t*(y1-y)
            cz = z + t*(z1-z)
            r = r0 + t*(r1-r0)
            r_ceil = math.ceil(r)
            for dx in range(-r_ceil, r_ceil+1):
                for dy in range(-r_ceil, r_ceil+1):
                    for dz in range(-r_ceil, r_ceil+1):
                        if dx*dx + dy*dy + dz*dz <= r*r:
                            vx = int(math.floor(cx + dx + 0.5))
                            vy = int(math.floor(cy + dy + 0.5))
                            vz = int(math.floor(cz + dz + 0.5))
                            if 0 <= vx < grid_size and 0 <= vy < grid_size and 0 <= vz < grid_size:
                                voxels[vx, vy, vz] = matWood

    # build trunk
    # start trunk at center of voxel grid so tree is centered in preview/export
    x = float(grid_size) / 2.0
    y = 0.0
    z = float(grid_size) / 2.0
    dx = 0.0; dy = 1.0; dz = 0.0
    for i in range(1, pTrunkIter+1):
        dx += rng.uniform(-bendVar, bendVar)
        dz += rng.uniform(-bendVar, bendVar)
        dx,dy,dz = normalize(dx, 1.0, dz)
        x1 = x + dx * segL
        y1 = y + dy * segL
        z1 = z + dz * segL
        t0 = (i-1)/pTrunkIter
        t1 = i/pTrunkIter
        r0 = radiusAt01(t0)
        r1 = radiusAt01(t1)
        line(x,y,z, x1,y1,z1, r0, r1)
        CX[i] = x1; CY[i] = y1; CZ[i] = z1; CR[i] = r1
        x,y,z = x1,y1,z1
    CX[pTrunkIter+1] = CX[pTrunkIter]
    CY[pTrunkIter+1] = CY[pTrunkIter]
    CZ[pTrunkIter+1] = CZ[pTrunkIter]
    CR[pTrunkIter+1] = CR[pTrunkIter]

    # core stitch
    for i in range(2, pTrunkIter+1):
        x0,y0,z0 = CX[i-1],CY[i-1],CZ[i-1]
        x1,y1,z1 = CX[i], CY[i], CZ[i]
        r0 = CR[i-1]*0.14
        r1 = CR[i]*0.14
        line(x0,y0,z0, x1,y1,z1, r0, r1)

    # flat base
    r = math.ceil(baseR * 1.25)
    for xx in range(-r, r+1):
        for zz in range(-r, r+1):
            vx = int(math.floor(xx + 0.5))
            vy = int(math.floor(0 + 0.5))
            vz = int(math.floor(zz + 0.5))
            if 0 <= vx < grid_size and 0 <= vy < grid_size and 0 <= vz < grid_size:
                voxels[vx, vy, vz] = 0

    # canopy anchor
    xt, yt, zt = CX[pTrunkIter], CY[pTrunkIter], CZ[pTrunkIter]
    xp, yp, zp = CX[pTrunkIter-1], CY[pTrunkIter-1], CZ[pTrunkIter-1]
    dx0, dy0, dz0 = normalize(xt - xp, yt - yp, zt - zp)
    cAnchorX, cAnchorY, cAnchorZ = xt, yt, zt
    cAnchorDX, cAnchorDY, cAnchorDZ = dx0, dy0, dz0
    rTop = CR[pTrunkIter]
    cJoinRadius = max(0.0, rTop - 0.8)
    gTrunkSizeBase = 6.0 * pTrunkSize * pSize
    cBaseScale = clamp((cJoinRadius) / (gTrunkSizeBase + 1e-6), 0.15, 4.0)

    # define helpers for roots and canopy
    def makeRootParams():
        def j(v): return (rng.uniform(-1.0,1.0) * v * pRootVariance)
        heightF = clamp(1.0 + j(0.40), 0.65, 1.25)
        spreadF = clamp(1.0 + j(0.50), 0.0, 1.50)
        thicknessF = clamp(1.0 + j(0.45), 0.25, 1.75)
        wedgeF = clamp(1.0 + j(0.35), 0.60, 1.50)
        ridgeF = clamp(1.0 + j(0.50), 0.40, 2.00)
        sinkStart = clamp(sinkStartBase + j(0.18), 0.05, 0.90)
        phaseJit = j(0.25)
        return {
            'rootHeight': clamp(rootHeightBase * heightF, 0.55, 0.985),
            'fanCap': max(0.0, fanCapBase * spreadF),
            'peakScale': thicknessF,
            'wedgeHalfA': clamp(wedgeHalfA_Base * wedgeF, 0.07, 0.35),
            'angSteps': max(12, math.floor(angStepsBase * (0.9 + 0.4*pRootVariance))),
            'ridgeAmpK': ridgeAmpK_Base * ridgeF,
            'sinkStart': sinkStart,
            'phaseJit': phaseJit
        }

    def sampleAtY_int(yf):
        return sampleAtY(yf)

    def drawRootWedge(theta0, P):
        yMaxFrac = clamp(P['rootHeight'], 0.0, 0.985)
        yMax = min(math.floor(H * yMaxFrac + 0.5), math.floor(H) - 1)
        twistAmount = pRootTwist * 1.3 * math.pi
        halfA = P['wedgeHalfA']
        steps = P['angSteps']
        ridgeK = P['ridgeAmpK']
        thinS = 0.12 + 0.88 * (pRootSpread ** 1.2)
        heightExp = 0.40
        for y in range(0, int(yMax)+1):
            tH = y / yMax if yMax != 0 else 0
            mRaw = (tH - P['sinkStart']) / (1.0 - P['sinkStart']) if (1.0 - P['sinkStart'])!=0 else 0
            u = smooth5(mRaw)
            holdW = 1.0 if u < 0.80 else (1.0 - smooth5((u - 0.80) / 0.20))
            thetaC = theta0 + tH * twistAmount
            cx, cz, rTrunk = sampleAtY_int(y + 0.35)
            off = (P['fanCap'] * (1.0 - (tH ** 2.0))) * (1.0 - u)
            peakH = (1.0 - (tH ** heightExp)) * rTrunk
            peak = (0.45 + 0.45 * pRootProfile) * peakH * (1.0 - u) * thinS * P['peakScale']
            seam = seamOverlapVox * (1.0 - 0.35 * tH)
            sink = u * rTrunk
            rIn = max(0.0, rTrunk - seam - sink)
            rSurf = rTrunk + off
            rHold = rTrunk - 1.0
            rSkin = rHold * holdW + (rSurf * (1.0 - u)) * (1.0 - holdW)
            for s in range(0, steps+1):
                w = (s / steps) * 2.0 - 1.0
                ang = thetaC + w * halfA
                tri = max(0.0, 1.0 - abs(w))
                ridge = (ridgeK * thinS) * (tri ** ridgeSharp) * (1.0 - tH) * (1.0 - u) * rTrunk
                thick = peak * tri + ridge
                rOut = (rSkin + thick) * (1.0 - u)
                if rOut < rIn: rOut = rIn
                ux = math.cos(ang); uz = math.sin(ang)
                for rr in range(math.floor(max(0.0, rIn - radialPad*(1.0 - u))), math.floor(rOut)+1):
                    xx = math.floor(cx + ux*rr + 0.5)
                    zz = math.floor(cz + uz*rr + 0.5)
                    if 0 <= xx < grid_size and 0 <= y < grid_size and 0 <= zz < grid_size:
                        if voxels[xx, y, zz] == 0:
                            voxels[xx, y, zz] = matWood
        for yy in range(max(0, int(yMax)-3), min(int(yMax)+1, math.floor(H))):
            cx, cz, _ = sampleAtY_int(yy + 0.5)
            xx = math.floor(cx + 0.5); zz = math.floor(cz + 0.5)
            if 0 <= xx < grid_size and 0 <= yy < grid_size and 0 <= zz < grid_size:
                voxels[xx, yy, zz] = matWood

    def fillWatertight():
        R = math.ceil(baseR + fanBase + 6)
        W = 2*R + 1
        # clamp vertical scan to the voxel grid to avoid indexing beyond bounds
        max_y = min(math.ceil(H), grid_size - 1)
        for y in range(0, max_y + 1):
             visited = set()
             q = []
             def key(ax,az): return (ax+R) + W*(az+R) + 1
             def seen(ax,az): return key(ax,az) in visited
             def push(ax,az): visited.add(key(ax,az)); q.append((ax,az))
             for x_ in range(-R, R+1):
                 if 0 <= (x_+R) < W and 0 <= (-R+R) < W:
                     if 0 <= x_ < grid_size and 0 <= y < grid_size and 0 <= -R < grid_size and voxels[x_, y, -R] == 0 and not seen(x_,-R):
                         push(x_,-R)
                 if 0 <= x_ < grid_size and 0 <= y < grid_size and 0 <= R < grid_size and voxels[x_, y, R] == 0 and not seen(x_,R):
                     push(x_,R)
             for z_ in range(-R, R+1):
                 if 0 <= -R < grid_size and 0 <= z_ < grid_size and voxels[-R, y, z_] == 0 and not seen(-R,z_):
                     push(-R,z_)
                 if 0 <= R < grid_size and 0 <= z_ < grid_size and voxels[R, y, z_] == 0 and not seen(R,z_):
                     push(R,z_)
             qh = 0
             while qh < len(q):
                 cx, cz = q[qh]; qh += 1
                 for nx, nz in ((cx-1, cz),(cx+1, cz),(cx, cz-1),(cx, cz+1)):
                     if nx >= -R and nx <= R and nz >= -R and nz <= R:
                         if 0 <= nx < grid_size and 0 <= y < grid_size and 0 <= nz < grid_size and voxels[nx, y, nz] == 0 and not seen(nx,nz):
                             push(nx,nz)
             for x_ in range(-R, R+1):
                 for z_ in range(-R, R+1):
                     if 0 <= x_ < grid_size and 0 <= y < grid_size and 0 <= z_ < grid_size:
                         if voxels[x_, y, z_] == 0 and key(x_,z_) not in visited:
                            voxels[x_, y, z_] = matWood

    # canopy derived
    cSize = 150 * pSize / pCanopyIter
    gTrunkSize = 6.0 * pTrunkSize * pSize
    cBranchLength0 = cSize * (1 - pWide)
    cBranchLength1 = cSize * pWide
    gLeaves = []

    def getFlatness(iter_):
        t = (iter_ - 1) / pCanopyIter
        t = math.sqrt(t)
        if t <= pCanopyStart or pCanopyFlat <= 0.0: return 0.0
        denom = (1.0 - pCanopyStart)
        if denom <= 0.0001: return 0.0
        u = clamp((t - pCanopyStart) / denom, 0.0, 1.0)
        return clamp(u * pCanopyFlat, 0.0, 1.0)

    def getBranchSize(iter_):
        t = (iter_ - 1) / pCanopyIter
        t = math.sqrt(t)
        alpha = 1.0 + 2.5 * pCanopyThick
        hold = 1.0 - (t ** alpha)
        return hold * gTrunkSize

    def getBranchLength(iter_):
        t = (iter_ - 1) / pCanopyIter
        t = math.sqrt(t)
        return cBranchLength0 + t * (cBranchLength1 - cBranchLength0)

    def getBranchAngle(iter_):
        t = (iter_ - 1) / pCanopyIter
        t = math.sqrt(t)
        return 2.0 * pSpread * t

    def getBranchProbability(iter_):
        t = (iter_ - 1) / pCanopyIter
        t = math.sqrt(t)
        base = t
        tierStrength = 0.15 + 0.55 * pCanopyProfile
        mu1, mu2, mu3 = 0.60, 0.78, 0.92
        s1 = 0.10 - 0.04 * pCanopyProfile
        s2 = 0.08 - 0.03 * pCanopyProfile
        s3 = 0.06 - 0.02 * pCanopyProfile
        a1, a2, a3 = 1.00, 0.75, 0.55
        def gauss(x,m,s): return math.exp(-((x-m)*(x-m))/(2.0*s*s))
        p = base * (1.0 - 0.5 * tierStrength)
        p = p + tierStrength * (a1*gauss(t,mu1,s1) + a2*gauss(t,mu2,s2) + a3*gauss(t,mu3,s3))
        return clamp(p, 0.0, 1.0)

    def generateBranches(x, y, z, dx, dy, dz, iter_):
        l = getBranchLength(iter_)
        s0 = getBranchSize(iter_) * cBaseScale
        s1 = getBranchSize(iter_ + 1) * cBaseScale
        if iter_ <= cJoinHoldSteps:
            tHold = (iter_ - 1) / max(1, cJoinHoldSteps)
            s0 = cJoinRadius * (1.0 - tHold) + s0 * tHold
            s1 = cJoinRadius * (1.0 - tHold) + s1 * tHold
        if pCanopyThick > 0.0:
            tiltGain = 0.25 * pCanopyThick
            down = max(0.0, -dy)
            boost = 1.0 + tiltGain * down
            s0 = s0 * boost
            s1 = s1 * boost
        x1 = x + dx * l
        y1 = y + dy * l
        z1 = z + dz * l
        line(x, y, z, x1, y1, z1, s0, s1)
        if iter_ < pCanopyIter:
            b = 1
            var = 1.0 * iter_ * 0.2 * pCanopyTwist
            if rng.uniform(0.0, 1.0) < getBranchProbability(iter_):
                b = 2
                var = getBranchAngle(iter_)
            x2, y2, z2 = x1, y1, z1
            for i in range(1, b+1):
                dx2 = dx + rng.uniform(-var, var)
                dy2 = dy + rng.uniform(-var, var)
                dz2 = dz + rng.uniform(-var, var)
                cx, cz, rTrunk = sampleAtY_int(y2)
                rx, rz = x2 - cx, z2 - cz
                orad = math.sqrt(rx*rx + rz*rz)
                if orad > 0.0001:
                    t = (iter_) / pCanopyIter
                    ow = 0.0
                    if t > pCanopyStart:
                        ow = ((t - pCanopyStart) / (1.0 - pCanopyStart)) * (0.35 * pCanopyProfile)
                    if ow > 0.0:
                        ux = rx / orad
                        uz = rz / orad
                        dx2 = dx2 + ux * ow
                        dz2 = dz2 + uz * ow
                w = getFlatness(iter_ + 1)
                dy2 = dy2 * (1.0 - w)
                blend = clamp((cJoinBlendSteps - (iter_ - 1)) / cJoinBlendSteps, 0.0, 1.0)
                dx2 = dx2 * (1.0 - blend) + cAnchorDX * blend
                dy2 = dy2 * (1.0 - blend) + cAnchorDY * blend
                dz2 = dz2 * (1.0 - blend) + cAnchorDZ * blend
                dx2, dy2, dz2 = normalize(dx2, dy2, dz2)
                generateBranches(x2, y2, z2, dx2, dy2, dz2, iter_ + 1)
        else:
            gLeaves.append([x1, y1, z1])
            gLeaves.append([(x + x1)/2, (y + y1)/2, (z + z1)/2])

    def generateLeaves():
        chunks = math.ceil(5 * pLeaves)
        leavesPerChunk = math.ceil(50 * pLeaves)
        baseVertProb = 2.0 / 6.0
        vertProb = baseVertProb * (1.0 - 0.85 * pCanopyProfile)
        horizProb = 1.0 - vertProb
        stepProb = horizProb / 4.0
        gravityAdj = clamp(pGravity - 0.35 * pCanopyProfile, -1.0, 1.0)
        leaf_positions = set()
        for pos in gLeaves:
            x1,y1,z1 = pos
            for _ in range(chunks):
                x2,y2,z2 = x1,y1,z1
                for _ in range(leavesPerChunk):
                    leaf_positions.add((int(x2), int(y2), int(z2)))
                    r = rng.uniform(0.0,1.0)
                    if r < stepProb:
                        x2 = x2 - 1
                    elif r < stepProb * 2.0:
                        x2 = x2 + 1
                    elif r < stepProb * 3.0:
                        z2 = z2 - 1
                    elif r < stepProb * 4.0:
                        z2 = z2 + 1
                    else:
                        if rng.uniform(-1.0, 1.0) > gravityAdj:
                            y2 = y2 - 1
                        else:
                            y2 = y2 + 1
        leaf_list = list(leaf_positions)
        rng.shuffle(leaf_list)
        for i, (x, y, z) in enumerate(leaf_list):
            idx = matLeaf
            if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
                if voxels[x, y, z] == 0:
                    voxels[x, y, z] = idx

    # shape 1 trunk + roots
    # build roots
    if pRootCount > 0:
        basePhase = rng.uniform(0, 2*math.pi)
        for k in range(1, pRootCount+1):
            P = makeRootParams()
            theta0 = basePhase + (k-1) * (2*math.pi / max(1,pRootCount)) + P['phaseJit'] * (2*math.pi / max(3, max(1,pRootCount)))
            drawRootWedge(theta0, P)
    fillWatertight()

    # canopy uses separate deterministic seed
    if pRandom != 0:
        rng.seed(pRandom + 1337)

    # shape 2 canopy
    gLeaves = []
    generateBranches(cAnchorX, cAnchorY, cAnchorZ, cAnchorDX, cAnchorDY, cAnchorDZ, 1)
    generateLeaves()

    # return voxels and palette
    exporter = VoxExporter(params, None, 'kapok', 'kapok')
    palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name)
    if preview:
        return voxels, palette

    # Map trunk and leaf placeholder voxels (matWood, matLeaf) into palette indices
    # Trunk: vertical banding similar to palm_worker
    trunk_coords = np.argwhere(voxels == matWood)
    trunk_list = [tuple(map(int, c)) for c in trunk_coords] if trunk_coords.size else []
    if trunk_list and trunk_indices:
        # Map all trunk voxels to random trunk palette indices (no strip/banding)
        trunk_coords_rand = np.argwhere(voxels == matWood)
        if trunk_coords_rand.size:
            for c in trunk_coords_rand:
                x, y, z = int(c[0]), int(c[1]), int(c[2])
                voxels[x, y, z] = rng.choice(trunk_indices)
    else:
        for i, (x, y, z) in enumerate(trunk_list):
            voxels[x, y, z] = trunk_indices[i % len(trunk_indices)] if trunk_indices else matWood

    # Leaves: map placeholder leaf voxels to leaf_indices using a simple vertical t01 proxy
    leaf_coords = np.argwhere(voxels == matLeaf)
    leaf_items = []
    if leaf_coords.size:
        ys = leaf_coords[:, 1]
        min_y = int(ys.min()); max_y = int(ys.max()); span = max(1, max_y - min_y)
        for x, y, z in map(tuple, leaf_coords):
            t01 = (y - min_y) / span
            leaf_items.append(((int(x), int(y), int(z)), t01))
    rng.shuffle(leaf_items)
    n_colors = len(leaf_indices) if leaf_indices else 0
    for (x, y, z), t in leaf_items:
        # skip if already overwritten by trunk mapping
        if voxels[x, y, z] != matLeaf and voxels[x, y, z] != 0:
            continue
        # Map leaves to random entries within the leaf palette range for a speckled look
        if n_colors:
            chosen = rng.choice(leaf_indices)
            voxels[x, y, z] = chosen
        else:
            voxels[x, y, z] = leaf_indices[0] if leaf_indices else matLeaf

    return voxels, palette


def generate_kapok_preview(params, palette_name, grid_size=PREVIEW_GRID, view='front', progress_callback=None, cancel_check=None):
    """
    Kapok preview wrapper that returns a PIL.Image when possible.
    Ensure Y/Z are swapped so the kapok preview shows the correct side/front orientation.
    """
    try:
        vox, palette = generate_kapok_tree(params, palette_name, grid_size=GRID, preview=False, progress_callback=progress_callback, cancel_check=cancel_check)
        # kapok uses same orientation as palm for MagicaVoxel preview/export -> swap Y/Z for UI preview
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
        vox, palette = generate_kapok_tree(params_preview, palette_name, grid_size=grid_size, preview=True, progress_callback=progress_callback, cancel_check=cancel_check)
        # kapok uses same orientation as palm for MagicaVoxel preview/export -> swap Y/Z for UI preview
        if isinstance(vox, np.ndarray):
            vox = np.swapaxes(vox, 1, 2)
        img = project_voxels_to_image(vox, palette, grid_size, view=view)
        return img.resize((grid_size * 3, grid_size * 3), Image.NEAREST)


def export_kapok(params, palette_name, prefix='kapok'):
    # Scale params to fit the tree within the 256 grid for export
    params_scaled = params.copy()
    scale_factor = 0.6  # Adjust to ensure max extent <= 256
    params_scaled['size'] = params.get('size', 1.0) * scale_factor
    params_scaled['trunkextend'] = params.get('trunkextend', 0.0) * scale_factor
    params_scaled['trunksize'] = params.get('trunksize', 1.0) * scale_factor
    # Generate full mapped voxel grid (ensure palette indices for leaves/trunk are applied)
    voxels, palette = generate_kapok_tree(params_scaled, palette_name, grid_size=GRID, preview=False)
    # Orient voxels for MagicaVoxel export by swapping Y and Z axes
    try:
        voxels_oriented = np.swapaxes(voxels, 1, 2).copy()
    except Exception:
        voxels_oriented = voxels
    exporter = VoxExporter(params_scaled, None, 'kapok', 'kapok')
    loaded_palette, leaf_indices, trunk_indices = exporter.load_palette(palette_name)
    return exporter.export(voxels_oriented, loaded_palette, leaf_indices, trunk_indices, prefix, preview=False)
