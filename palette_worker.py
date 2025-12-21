"""palette_worker.py

Internal palette manager to avoid relying on external PNG palette files.
- Provides built-in palettes (default, pine, autumn) generated procedurally.
- Exposes an API to retrieve a 256-length RGBA palette and semantic index mappings
  (for example: which palette indices should be considered `leaves` or `trunk`).

This module is intentionally conservative: palettes are generated programmatically
so there is no dependency on external image files. Mapping keys and default index
ranges follow the conventions used elsewhere in the project (e.g. leaves indices
9..17, trunk indices 57..65) but can be overridden at registration time.
"""
from typing import List, Tuple, Dict, Optional

RGBA = Tuple[int, int, int, int]


def _clamp8(v: int) -> int:
    return max(0, min(255, int(v)))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _gradient(start: RGBA, end: RGBA, count: int) -> List[RGBA]:
    """Return a list of `count` colors from start to end (inclusive of start).
    If count <= 0 returns empty list.
    """
    if count <= 0:
        return []
    if count == 1:
        return [start]
    out: List[RGBA] = []
    for i in range(count):
        t = i / (count - 1)
        r = _clamp8(round(_lerp(start[0], end[0], t)))
        g = _clamp8(round(_lerp(start[1], end[1], t)))
        b = _clamp8(round(_lerp(start[2], end[2], t)))
        a = _clamp8(round(_lerp(start[3], end[3], t)))
        out.append((r, g, b, a))
    return out


class InternalPaletteManager:
    """Manage internal 256-color palettes and semantic index maps.

    Usage:
    - get_palette(name) -> (palette_list, map_config)
    - register_palette(name, palette_256, map_config)

    map_config is a dict that can contain keys such as 'leaves' and 'trunk',
    each mapping to either a list of indices or a pair [start, end] inclusive.
    """

    def __init__(self):
        self._palettes: Dict[str, List[RGBA]] = {}
        self._maps: Dict[str, Dict[str, List[int]]] = {}
        self._build_defaults()

    def _ensure_256(self, palette: List[RGBA]) -> List[RGBA]:
        if len(palette) >= 256:
            return palette[:256]
        # pad with transparent black or repeated last color
        pad = [(0, 0, 0, 0)] * (256 - len(palette))
        return palette + pad

    def register_palette(self, name: str, palette: List[RGBA], map_config: Optional[Dict[str, Tuple[int, int] or List[int]]] = None):
        """Register a new palette and optional semantic map.

        map_config values may be either a 2-tuple indicating an inclusive range
        or an explicit list of indices.
        """
        p = self._ensure_256(palette)
        self._palettes[name] = p
        # Normalize map_config to lists of ints
        norm: Dict[str, List[int]] = {}
        if map_config:
            for k, v in map_config.items():
                if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, int) for x in v):
                    start, end = v
                    norm[k] = list(range(start, end + 1))
                elif isinstance(v, list) and all(isinstance(x, int) for x in v):
                    norm[k] = v[:]
                else:
                    # ignore invalid entries
                    continue
        self._maps[name] = norm

    def get_palette(self, name: Optional[str] = None) -> (List[RGBA], Dict[str, List[int]]):
        """Return (palette, map) for `name`. If not found, returns 'default'.
        If name is None returns the 'default' palette.
        """
        key = name or 'default'
        if key not in self._palettes:
            key = 'default'
        return self._palettes[key], self._maps.get(key, {})

    def list_palettes(self) -> List[str]:
        return sorted(self._palettes.keys())

    # --- default palette construction ---
    def _build_defaults(self):
        # Default: grayscale ramp
        gray = [(i, i, i, 255) for i in range(256)]
        self.register_palette('default', gray, map_config={'leaves': [9, 17], 'trunk': [57, 65]})

        # -- Pine family: multiple procedural variants --
        # Pine default: classic deep greens, row-aligned for MagicaVoxel
        pine_default = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step dark green needle gradient around #334A36
        leaf_grad = [
            (0x28, 0x3C, 0x28, 255),  # darker shade
            (0x2D, 0x42, 0x2D, 255),
            (0x32, 0x48, 0x32, 255),
            (0x33, 0x4A, 0x36, 255),  # around #334A36
            (0x38, 0x50, 0x3B, 255),
            (0x3D, 0x56, 0x40, 255),
            (0x42, 0x5C, 0x45, 255),
            (0x47, 0x62, 0x4A, 255),  # lighter shade
        ]
        pine_default[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step brown bark gradient
        trunk_grad = [
            (0x7E, 0x77, 0x5A, 255),  # index 57: 7e775a
            (0x7B, 0x74, 0x57, 255),  # index 58: 7b7457
            (0x78, 0x71, 0x55, 255),  # index 59: 787155
            (0x75, 0x6E, 0x53, 255),  # index 60: 756e53
            (0x72, 0x6B, 0x51, 255),  # index 61: 726b51
            (0x6F, 0x68, 0x4F, 255),  # index 62: 6f684f
            (0x6C, 0x65, 0x4D, 255),  # index 63: 6c654d
            (0x69, 0x63, 0x4B, 255),  # index 64: 69634b
        ]
        pine_default[57:64] = trunk_grad

        # Fill other rows with pine-inspired gradients
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                pine_default[i] = (
                    int(18 + 102 * t),
                    int(36 + 76 * t),
                    int(12 + 38 * t),
                    255
                )
        self.register_palette('Default Pine', pine_default, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Pine basic: slightly brighter leaves and thinner trunk band, row-aligned for MagicaVoxel
        pine_basic = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - use requested colors
        leaf_grad = [
            (0x4F, 0x61, 0x4B, 255),  # index 9: 4f614b
            (0x51, 0x63, 0x4D, 255),  # index 10: 51634d
            (0x53, 0x66, 0x4F, 255),  # index 11: 53664f
            (0x56, 0x69, 0x51, 255),  # index 12: 566951
            (0x58, 0x6C, 0x53, 255),  # index 13: 586c53
            (0x5A, 0x6F, 0x55, 255),  # index 14: 5a6f55
            (0x5D, 0x72, 0x58, 255),  # index 15: 5d7258
            (0x49, 0x5A, 0x46, 255),  # index 16: 495a46
        ]
        pine_basic[9:17] = leaf_grad

        # Row 8: indices 57-64 (trunk) - use requested trunk colors
        trunk_grad = [
            (0x7E, 0x77, 0x5A, 255),  # index 57: 7e775a
            (0x7B, 0x74, 0x57, 255),  # index 58: 7b7457
            (0x78, 0x71, 0x55, 255),  # index 59: 787155
            (0x75, 0x6E, 0x53, 255),  # index 60: 756e53
            (0x72, 0x6B, 0x51, 255),  # index 61: 726b51
            (0x6F, 0x68, 0x4F, 255),  # index 62: 6f684f
            (0x6C, 0x65, 0x4D, 255),  # index 63: 6c654d
            (0x69, 0x63, 0x4B, 255),  # index 64: 69634b
        ]
        pine_basic[57:65] = trunk_grad

        # Fill other rows with green/brown gradients
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                pine_basic[i] = (
                    int(30 + 100 * t),
                    int(60 + 80 * t),
                    int(20 + 40 * t),
                    255
                )
        self.register_palette('Basic Pine', pine_basic, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Redpine / repine: warmer reddish needles, row-aligned for MagicaVoxel
        redpine = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step reddish needle gradient around #4d312f
        leaf_grad = [
            (0x5b, 0x65, 0x4b, 255),  # index 9
            (0x57, 0x61, 0x48, 255),  # index 10
            (0x53, 0x5d, 0x45, 255),  # index 11
            (0x50, 0x59, 0x42, 255),  # index 12
            (0x4c, 0x55, 0x3f, 255),  # index 13
            (0x49, 0x51, 0x3c, 255),  # index 14
            (0x45, 0x4d, 0x39, 255),  # index 15
            (0x42, 0x4a, 0x37, 255),  # index 16
        ]
        redpine[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step dark brown bark gradient
        trunk_grad = [
            (0x5f, 0x4e, 0x43, 255),  # index 57
            (0x57, 0x47, 0x3e, 255),  # index 58
            (0x4d, 0x3c, 0x36, 255),  # index 59
            (0x45, 0x37, 0x31, 255),  # index 60
            (0x43, 0x33, 0x2e, 255),  # index 61
            (0x3b, 0x2d, 0x29, 255),  # index 62
            (0x40, 0x2d, 0x2b, 255),  # index 63
            (0x4d, 0x36, 0x34, 255),  # index 64
        ]
        redpine[57:64] = trunk_grad

        # Fill other rows with reddish gradients
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                redpine[i] = (
                    int(120 + 80 * t),
                    int(40 + 30 * t),
                    int(20 + 30 * t),
                    255
                )
        self.register_palette('Red Pine', redpine, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Pine sapling: small bright leaf band and lighter trunk, row-aligned for MagicaVoxel
        pine_sapling = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step bright green needle gradient around #91b124
        leaf_grad = [
            (0x78, 0x96, 0x24, 255),  # darker shade
            (0x80, 0xA0, 0x28, 255),
            (0x88, 0xAA, 0x2C, 255),
            (0x91, 0xB1, 0x24, 255),  # around #91b124
            (0x9A, 0xBA, 0x28, 255),
            (0xA3, 0xC3, 0x2C, 255),
            (0xAC, 0xCC, 0x30, 255),
            (0xB5, 0xD5, 0x34, 255),  # lighter shade
        ]
        pine_sapling[9:17] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step lighter brown bark gradient
        trunk_grad = [
            (0x64, 0x46, 0x32, 255),  # lighter sapling bark
            (0x6E, 0x50, 0x38, 255),
            (0x78, 0x5A, 0x3E, 255),
            (0x82, 0x64, 0x44, 255),
            (0x8C, 0x6E, 0x4A, 255),
            (0x96, 0x78, 0x50, 255),
            (0xA0, 0x82, 0x56, 255),
            (0xAA, 0x8C, 0x5C, 255),
        ]
        pine_sapling[57:65] = trunk_grad

        # Fill other rows with bright green gradients
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                pine_sapling[i] = (
                    int(80 + 50 * t),
                    int(160 + 40 * t),
                    int(60 + 20 * t),
                    255
                )
        self.register_palette('Pine Sapling', pine_sapling, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Scots pine (scotspine): muted, cool greens with darker trunk, row-aligned for MagicaVoxel
        scotspine = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - use requested colors
        leaf_grad = [
            (0x5B, 0x65, 0x4B, 255),  # index 9: 5b654b
            (0x57, 0x61, 0x48, 255),  # index 10: 576148
            (0x53, 0x5D, 0x45, 255),  # index 11: 535d45
            (0x50, 0x59, 0x42, 255),  # index 12: 505942
            (0x4C, 0x55, 0x3F, 255),  # index 13: 4c553f
            (0x49, 0x51, 0x3C, 255),  # index 14: 49513c
            (0x45, 0x4D, 0x39, 255),  # index 15: 454d39
            (0x42, 0x4A, 0x37, 255),  # index 16: 424a37
        ]
        scotspine[9:17] = leaf_grad

        # Row 8: indices 57-64 (trunk) - use requested trunk colors
        trunk_grad = [
            (0xB4, 0x9C, 0x80, 255),  # index 57: b49c80
            (0xAD, 0x96, 0x7B, 255),  # index 58: ad967b
            (0xA6, 0x90, 0x76, 255),  # index 59: a69076
            (0x9F, 0x8A, 0x71, 255),  # index 60: 9f8a71
            (0x99, 0x84, 0x6C, 255),  # index 61: 99846c
            (0x92, 0x7E, 0x67, 255),  # index 62: 927e67
            (0x8B, 0x78, 0x62, 255),  # index 63: 8b7862
            (0x85, 0x72, 0x5E, 255),  # index 64: 85725e
        ]
        scotspine[57:65] = trunk_grad

        # Fill other rows with cool green gradients
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                scotspine[i] = (
                    int(25 + 85 * t),
                    int(50 + 50 * t),
                    int(20 + 30 * t),
                    255
                )
        self.register_palette('Scots Pine', scotspine, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Autumn palette: emphasis on reds/oranges for leaves and darker trunks, row-aligned for MagicaVoxel
        autumn = [(0x0a, 0x0a, 0x0a, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step red/orange leaf gradient
        leaf_grad = [
            (0x78, 0x28, 0x14, 255),  # deep red-orange
            (0x8C, 0x32, 0x18, 255),
            (0xA0, 0x3C, 0x1C, 255),
            (0xB4, 0x46, 0x20, 255),
            (0xC8, 0x50, 0x24, 255),
            (0xDC, 0x5A, 0x28, 255),
            (0xF0, 0x64, 0x2C, 255),
            (0xFF, 0x6E, 0x30, 255),
        ]
        autumn[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step darker brown trunk gradient
        trunk_grad = [
            (0x58, 0x4D, 0x43, 255),  # index 57
            (0x5B, 0x4F, 0x44, 255),  # index 58
            (0x5F, 0x52, 0x46, 255),  # index 59
            (0x63, 0x55, 0x48, 255),  # index 60
            (0x66, 0x58, 0x4A, 255),  # index 61
            (0x6A, 0x5B, 0x4C, 255),  # index 62
            (0x6E, 0x5E, 0x4E, 255),  # index 63
            (0x72, 0x61, 0x50, 255),  # index 64
        ]
        autumn[57:64] = trunk_grad

        # Fill other rows with a reddish gradient, but do not overwrite leaf/trunk rows
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue  # skip leaf/trunk rows
            for i in range(start, end):
                if i >= 256:
                    break
                # Use a reddish gradient for other rows
                t = (i - 1) / 255
                autumn[i] = (
                    int(0x78 + 0x64 * t),
                    int(0x28 + 0x28 * t),
                    int(0x14 + 0x14 * t),
                    255
                )
        self.register_palette('Autumn 1', autumn, map_config={'leaves': list(range(9, 16)), 'trunk': list(range(57, 64))})

        # Autumn 2: user-specified colors for leaves and trunk, row-aligned for MagicaVoxel
        autumn2 = [(10, 10, 10, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step autumn leaf gradient
        leaf_grad = [
            (0xA0, 0x50, 0x20, 255),  # warm orange-red
            (0xB0, 0x60, 0x30, 255),
            (0xC0, 0x70, 0x40, 255),
            (0xD0, 0x80, 0x50, 255),
            (0xE0, 0x90, 0x60, 255),
            (0xF0, 0xA0, 0x70, 255),
            (0xFF, 0xB0, 0x80, 255),
            (0xFF, 0xC0, 0x90, 255),
        ]
        autumn2[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step dark trunk gradient
        trunk_grad = [
            (0x58, 0x4D, 0x43, 255),  # index 57
            (0x5B, 0x4F, 0x44, 255),  # index 58
            (0x5F, 0x52, 0x46, 255),  # index 59
            (0x63, 0x55, 0x48, 255),  # index 60
            (0x66, 0x58, 0x4A, 255),  # index 61
            (0x6A, 0x5B, 0x4C, 255),  # index 62: 6a5b4c
            (0x6E, 0x5E, 0x4E, 255),  # index 63: 6e5e4e
            (0x72, 0x61, 0x50, 255),  # index 64
        ]
        autumn2[57:64] = trunk_grad

        # Fill other rows with a brownish gradient, but do not overwrite leaf/trunk rows
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue  # skip leaf/trunk rows
            for i in range(start, end):
                if i >= 256:
                    break
                # Use a brownish gradient for other rows
                t = (i - 1) / 255
                autumn2[i] = (
                    int(120 + 80 * t),
                    int(80 + 60 * t),
                    int(40 + 40 * t),
                    255
                )
        self.register_palette('Autumn 2', autumn2, map_config={'leaves': list(range(9, 16)), 'trunk': list(range(57, 64))})

        # Birch: pale leaves and light trunk
        birch = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step pale leaf gradient
        leaf_grad = [
            (0x6A, 0x74, 0x4A, 255),  # muted pale olive
            (0x7A, 0x86, 0x59, 255),
            (0x8C, 0x98, 0x68, 255),
            (0x9E, 0xAA, 0x77, 255),
            (0xB0, 0xBC, 0x86, 255),
            (0xC2, 0xCE, 0x96, 255),
            (0xD4, 0xE0, 0xA5, 255),
            (0xE6, 0xF2, 0xB4, 255),
        ]
        birch[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step pale bark gradient around #dbd2ad
        trunk_grad = [
            (0xB4, 0xAA, 0x8C, 255),  # darker shade
            (0xBE, 0xB4, 0x96, 255),
            (0xC8, 0xBE, 0xA0, 255),
            (0xDB, 0xD2, 0xAD, 255),  # around #dbd2ad
            (0xE0, 0xD7, 0xB2, 255),
            (0xE5, 0xDC, 0xB7, 255),
            (0xEA, 0xE1, 0xBC, 255),
            (0xEF, 0xE6, 0xC1, 255),  # lighter shade
        ]
        birch[57:64] = trunk_grad

        # Fill other rows with pale gradients (your original structure)
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                birch[i] = (
                    int(180 + 40 * t),
                    int(200 + 30 * t),
                    int(140 + 30 * t),
                    255
                )

        self.register_palette('Birch Variant 1', birch,
                              map_config={'leaves': [9, 16], 'trunk': [57, 64]})


        # Birch variants: white birch (paler bark), yellow birch (warm yellow leaves), gray birch (muted tones)
        # White birch: very light trunk with subtle darker flecks, leaves pale green, row-aligned for MagicaVoxel
        birch_white = [(6, 6, 6, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step pale green gradient
        leaf_grad = [
            (0x70, 0x82, 0x5A, 255),
            (0x80, 0x93, 0x67, 255),
            (0x90, 0xA4, 0x74, 255),
            (0xA0, 0xB5, 0x81, 255),
            (0xB3, 0xC6, 0x8F, 255),
            (0xC6, 0xD7, 0x9D, 255),
            (0xD8, 0xE7, 0xAB, 255),
            (0xEA, 0xF7, 0xB9, 255),
        ]
        birch_white[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step white-bark gradient with darker flecks
        trunk_grad = [
            (0xF0, 0xF0, 0xEB, 255),  # bright white bark tone
            (0xE6, 0xE4, 0xDF, 255),
            (0xDB, 0xD9, 0xD4, 255),
            (0xD0, 0xCE, 0xC9, 255),
            (0xC5, 0xC3, 0xBE, 255),
            (0xBA, 0xB8, 0xB3, 255),
            (0xAF, 0xAD, 0xA8, 255),
            (0xA4, 0xA2, 0x9D, 255),  # “fleck” shade
        ]
        birch_white[57:64] = trunk_grad

        # Fill other rows with light gradients (your structure)
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                birch_white[i] = (
                    int(190 + 60 * t),
                    int(210 + 30 * t),
                    int(170 + 30 * t),
                    255
                )

        self.register_palette('White Birch', birch_white,
                              map_config={'leaves': [9, 16], 'trunk': [57, 64]})


        # Yellow birch: warm yellow/orange leaves (late-summer/autumn look), warm trunk, row-aligned for MagicaVoxel
        birch_yellow = [(6, 6, 6, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step leaf gradient
        leaf_grad = [
            (0x4F, 0x4A, 0x16, 255),
            (0x6B, 0x66, 0x20, 255),
            (0x86, 0x7F, 0x1F, 255),
            (0xA7, 0x9C, 0x22, 255),
            (0xC4, 0xB7, 0x2A, 255),
            (0xDC, 0xC8, 0x3A, 255),
            (0xE9, 0xD9, 0x57, 255),
            (0xF7, 0xEF, 0xA0, 255),
        ]
        birch_yellow[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step trunk gradient (lightened darker shades)
        trunk_grad = [
            (0xF0, 0xF0, 0xEB, 255),  # bright white bark tone
            (0xE6, 0xE4, 0xDF, 255),
            (0xDB, 0xD9, 0xD4, 255),
            (0xD0, 0xCE, 0xC9, 255),
            (0xC5, 0xC3, 0xBE, 255),
            (0xBA, 0xB8, 0xB3, 255),
            (0xAF, 0xAD, 0xA8, 255),
            (0xA4, 0xA2, 0x9D, 255),  # “fleck” shade
        ]
        birch_yellow[57:64] = trunk_grad

        # Fill other rows with warm gradients (same as your original)
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                birch_yellow[i] = (
                    int(200 + 40 * t),
                    int(170 + 30 * t),
                    int(70 + 50 * t),
                    255
                )

        self.register_palette('Yellow Birch', birch_yellow,
                              map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Gray birch: muted, cool leaves and grayer trunk, row-aligned for MagicaVoxel
        birch_gray = [(6, 6, 6, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step cool muted leaf gradient
        leaf_grad = [
            (0x5C, 0x63, 0x54, 255),
            (0x68, 0x6D, 0x5E, 255),
            (0x74, 0x78, 0x68, 255),
            (0x80, 0x83, 0x72, 255),
            (0x8C, 0x8E, 0x7C, 255),
            (0x98, 0x99, 0x86, 255),
            (0xA4, 0xA4, 0x90, 255),
            (0xB0, 0xAF, 0x9A, 255),
        ]
        birch_gray[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step gray trunk gradient
        trunk_grad = [
            (0xC8, 0xC0, 0xB8, 255),
            (0xBC, 0xB4, 0xAC, 255),
            (0xB1, 0xA8, 0xA1, 255),
            (0xA6, 0x9D, 0x96, 255),
            (0x9B, 0x92, 0x8B, 255),
            (0x90, 0x87, 0x80, 255),
            (0x85, 0x7C, 0x75, 255),
            (0x7A, 0x71, 0x6A, 255),
        ]
        birch_gray[57:64] = trunk_grad

        # Fill other rows with gray gradients (your structure)
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                birch_gray[i] = (
                    int(150 + 50 * t),
                    int(170 + 30 * t),
                    int(140 + 30 * t),
                    255
                )

        self.register_palette('Gray Birch', birch_gray,
                              map_config={'leaves': [9, 16], 'trunk': [57, 64]})


        # Blossom: pink blossoms with warm trunk, row-aligned for MagicaVoxel
        blossom = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step pink blossom gradient
        leaf_grad = [
            (0xD8, 0x9E, 0xB4, 255),
            (0xE3, 0xA9, 0xBE, 255),
            (0xEE, 0xB4, 0xC8, 255),
            (0xF9, 0xBF, 0xD2, 255),
            (0xFF, 0xC9, 0xDA, 255),
            (0xFF, 0xD3, 0xE2, 255),
            (0xFF, 0xDD, 0xEA, 255),
            (0xFF, 0xE7, 0xF2, 255),
        ]
        blossom[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step warm bark with slight red tint gradient
        trunk_grad = [
            (0x60, 0x4D, 0x43, 255),  # index 57
            (0x63, 0x4F, 0x44, 255),  # index 58
            (0x67, 0x52, 0x46, 255),  # index 59
            (0x6B, 0x55, 0x48, 255),  # index 60
            (0x6E, 0x58, 0x4A, 255),  # index 61
            (0x72, 0x5B, 0x4C, 255),  # index 62
            (0x76, 0x5E, 0x4E, 255),  # index 63
            (0x7A, 0x61, 0x50, 255),  # index 64
        ]
        blossom[57:64] = trunk_grad

        # Fill other rows with pink gradients (your structure)
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                blossom[i] = (
                    int(220 + 35 * t),
                    int(160 + 40 * t),
                    int(180 + 40 * t),
                    255
                )

        self.register_palette('Blossom, Cherry', blossom,
                              map_config={'leaves': [9, 16], 'trunk': [57, 64]})


        # Blossom 2: user-specified blossom palette (indices provided by user)
        blossom2 = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - explicit colors provided by user
        leaf_grad = [
            (0x63, 0x9A, 0x39, 255),  # index 9
            (0x8F, 0x83, 0x50, 255),  # index 10
            (0xBB, 0x6C, 0x67, 255),  # index 11
            (0xE7, 0x55, 0x7E, 255),  # index 12
            (0xDF, 0x83, 0xA3, 255),  # index 13
            (0xD7, 0xB1, 0xC7, 255),  # index 14
            (0xC5, 0x7C, 0x9E, 255),  # index 15
            (0xB3, 0x46, 0x75, 255),  # index 16
        ]
        blossom2[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - explicit colors provided by user
        trunk_grad = [
            (0x58, 0x4D, 0x43, 255),  # index 57
            (0x5B, 0x4F, 0x44, 255),  # index 58
            (0x5F, 0x52, 0x46, 255),  # index 59
            (0x63, 0x55, 0x48, 255),  # index 60
            (0x66, 0x58, 0x4A, 255),  # index 61
            (0x6A, 0x5B, 0x4C, 255),  # index 62
            (0x6E, 0x5E, 0x4E, 255),  # index 63
            (0x72, 0x61, 0x50, 255),  # index 64
        ]
        blossom2[57:64] = trunk_grad

        # Fill other rows with a blossom-like pink gradient (do not overwrite leaf/trunk rows)
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                blossom2[i] = (
                    int(220 + 35 * t),
                    int(160 + 40 * t),
                    int(180 + 40 * t),
                    255
                )

        self.register_palette('Blossom, Apple', blossom2,
                              map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Dead: desaturated browns / grays, row-aligned for MagicaVoxel
        dead = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step dead leaf gradient
        leaf_grad = [
            (0x54, 0x4C, 0x46, 255),
            (0x5D, 0x55, 0x4F, 255),
            (0x66, 0x5E, 0x58, 255),
            (0x6F, 0x67, 0x61, 255),
            (0x78, 0x70, 0x6A, 255),
            (0x81, 0x79, 0x73, 255),
            (0x8A, 0x82, 0x7C, 255),
            (0x93, 0x8B, 0x85, 255),
        ]
        dead[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step dry wood gradient
        trunk_grad = [
            (0x58, 0x4D, 0x43, 255),  # index 57
            (0x5B, 0x4F, 0x44, 255),  # index 58
            (0x5F, 0x52, 0x46, 255),  # index 59
            (0x63, 0x55, 0x48, 255),  # index 60
            (0x66, 0x58, 0x4A, 255),  # index 61
            (0x6A, 0x5B, 0x4C, 255),  # index 62
            (0x6E, 0x5E, 0x4E, 255),  # index 63
            (0x72, 0x61, 0x50, 255),  # index 64
        ]
        dead[57:64] = trunk_grad

        # Fill other rows with gray-brown gradients
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                dead[i] = (
                    int(120 + 80 * t),
                    int(110 + 60 * t),
                    int(100 + 60 * t),
                    255
                )

        self.register_palette('Dead 1', dead,
                              map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Dead 2: user-specified desaturated brown/gray palette
        dead2 = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - explicit colors provided by user
        leaf_grad = [
            (0xA0, 0x65, 0x47, 255),  # index 9
            (0x9A, 0x66, 0x47, 255),  # index 10
            (0x94, 0x66, 0x46, 255),  # index 11
            (0x8E, 0x67, 0x46, 255),  # index 12
            (0x89, 0x67, 0x46, 255),  # index 13
            (0x83, 0x68, 0x46, 255),  # index 14
            (0x7D, 0x68, 0x45, 255),  # index 15
            (0x77, 0x69, 0x45, 255),  # index 16
        ]
        dead2[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - explicit colors provided by user
        trunk_grad = [
            (0x58, 0x4D, 0x43, 255),  # index 57
            (0x5B, 0x4F, 0x44, 255),  # index 58
            (0x5F, 0x52, 0x46, 255),  # index 59
            (0x63, 0x55, 0x48, 255),  # index 60
            (0x66, 0x58, 0x4A, 255),  # index 61
            (0x6A, 0x5B, 0x4C, 255),  # index 62
            (0x6E, 0x5E, 0x4E, 255),  # index 63
            (0x72, 0x61, 0x50, 255),  # index 64
        ]
        dead2[57:64] = trunk_grad

        # Fill other rows with gray-brown gradients (do not overwrite leaf/trunk rows)
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                dead2[i] = (
                    int(120 + 80 * t),
                    int(110 + 60 * t),
                    int(100 + 60 * t),
                    255
                )

        self.register_palette('Dead 2', dead2,
                              map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Oak1: richer greens, row-aligned for MagicaVoxel
        oak1 = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step green leaf gradient
        leaf_grad = [
            (0x28, 0x50, 0x14, 255),
            (0x30, 0x5A, 0x18, 255),
            (0x38, 0x64, 0x1C, 255),
            (0x40, 0x6E, 0x20, 255),
            (0x48, 0x78, 0x24, 255),
            (0x50, 0x82, 0x28, 255),
            (0x58, 0x8C, 0x2C, 255),
            (0x60, 0x96, 0x30, 255),
        ]
        oak1[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step brown trunk gradient
        trunk_grad = [
            (0x58, 0x4D, 0x43, 255),  # index 57
            (0x5B, 0x4F, 0x44, 255),  # index 58
            (0x5F, 0x52, 0x46, 255),  # index 59
            (0x63, 0x55, 0x48, 255),  # index 60
            (0x66, 0x58, 0x4A, 255),  # index 61
            (0x6A, 0x5B, 0x4C, 255),  # index 62
            (0x6E, 0x5E, 0x4E, 255),  # index 63
            (0x72, 0x61, 0x50, 255),  # index 64
        ]
        oak1[57:64] = trunk_grad

        # Fill other rows with green gradients
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                oak1[i] = (
                    int(40 + 90 * t),
                    int(80 + 60 * t),
                    int(20 + 40 * t),
                    255
                )

        self.register_palette('Oak Variant 1', oak1, map_config={'leaves': [9, 16], 'trunk': [57, 64]})


        # Oak2: richer greens, row-aligned for MagicaVoxel
        oak2 = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step green leaf gradient
        leaf_grad = [
            (0x1E, 0x46, 0x1E, 255),
            (0x28, 0x50, 0x24, 255),
            (0x32, 0x5A, 0x2A, 255),
            (0x3C, 0x64, 0x30, 255),
            (0x46, 0x6E, 0x36, 255),
            (0x50, 0x78, 0x3C, 255),
            (0x5A, 0x82, 0x42, 255),
            (0x46, 0x8C, 0x48, 255),  # slightly brighter top
        ]
        oak2[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step brown trunk gradient
        trunk_grad = [
            (0x58, 0x4D, 0x43, 255),  # index 57
            (0x5B, 0x4F, 0x44, 255),  # index 58
            (0x5F, 0x52, 0x46, 255),  # index 59
            (0x63, 0x55, 0x48, 255),  # index 60
            (0x66, 0x58, 0x4A, 255),  # index 61
            (0x6A, 0x5B, 0x4C, 255),  # index 62
            (0x6E, 0x5E, 0x4E, 255),  # index 63
            (0x72, 0x61, 0x50, 255),  # index 64
        ]
        oak2[57:64] = trunk_grad

        # Fill other rows with green gradients
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                oak2[i] = (
                    int(30 + 110 * t),
                    int(70 + 60 * t),
                    int(30 + 40 * t),
                    255
                )

        self.register_palette('Oak Variant 2', oak2, map_config={'leaves': [9, 16], 'trunk': [57, 64]})


        # Tree basic / default / sapling, row-aligned for MagicaVoxel
        tree_basic = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step green leaf gradient
        leaf_grad = [
            (0x32, 0x5A, 0x1E, 255),
            (0x3C, 0x64, 0x24, 255),
            (0x46, 0x6E, 0x28, 255),
            (0x50, 0x78, 0x2C, 255),
            (0x5A, 0x82, 0x30, 255),
            (0x64, 0x8C, 0x34, 255),
            (0x6E, 0x96, 0x38, 255),
            (0x78, 0xA0, 0x3C, 255),
        ]
        tree_basic[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step brown trunk gradient
        trunk_grad = [
            (0x58, 0x4D, 0x43, 255),  # index 57
            (0x5B, 0x4F, 0x44, 255),  # index 58
            (0x5F, 0x52, 0x46, 255),  # index 59
            (0x63, 0x55, 0x48, 255),  # index 60
            (0x66, 0x58, 0x4A, 255),  # index 61
            (0x6A, 0x5B, 0x4C, 255),  # index 62
            (0x6E, 0x5E, 0x4E, 255),  # index 63
            (0x72, 0x61, 0x50, 255),  # index 64
        ]
        tree_basic[57:64] = trunk_grad

        # Fill other rows with green gradients
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                tree_basic[i] = (
                    int(50 + 90 * t),
                    int(90 + 60 * t),
                    int(30 + 40 * t),
                    255
                )

        self.register_palette('Basic', tree_basic, map_config={'leaves': [9, 16], 'trunk': [57, 64]})


        # Tree sapling: bright green leaves, young trunk, row-aligned for MagicaVoxel
        tree_sapling = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step green leaf gradient
        leaf_grad = [
            (0x50, 0xA0, 0x3C, 255),
            (0x58, 0xAC, 0x3E, 255),
            (0x60, 0xB4, 0x40, 255),
            (0x68, 0xBC, 0x42, 255),
            (0x70, 0xC4, 0x44, 255),
            (0x78, 0xCC, 0x46, 255),
            (0x80, 0xD4, 0x48, 255),
            (0x88, 0xDC, 0x4A, 255),
        ]
        tree_sapling[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step brighter brown trunk gradient
        trunk_grad = [
            (0x60, 0x55, 0x4B, 255),  # index 57
            (0x63, 0x57, 0x4C, 255),  # index 58
            (0x67, 0x5A, 0x4E, 255),  # index 59
            (0x6B, 0x5D, 0x50, 255),  # index 60
            (0x6E, 0x60, 0x52, 255),  # index 61
            (0x72, 0x63, 0x54, 255),  # index 62
            (0x76, 0x66, 0x56, 255),  # index 63
            (0x7A, 0x69, 0x58, 255),  # index 64
        ]
        tree_sapling[57:64] = trunk_grad

        # Fill other rows with bright green gradients
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                tree_sapling[i] = (
                    int(80 + 50 * t),
                    int(160 + 40 * t),
                    int(60 + 20 * t),
                    255
                )
        self.register_palette('Sapling', tree_sapling, map_config={'leaves': [9, 16], 'trunk': [57, 64]})


        # Birch 2: pale leaves and light trunk, row-aligned for MagicaVoxel
        birch2 = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - 8-step leaf gradient
        leaf_grad = [
            (0x8A, 0xBE, 0x67, 255),
            (0x83, 0xB5, 0x61, 255),
            (0x7C, 0xAC, 0x5C, 255),
            (0x75, 0xA3, 0x57, 255),
            (0x6E, 0x9A, 0x52, 255),
            (0x67, 0x91, 0x4D, 255),
            (0x60, 0x88, 0x48, 255),
            (0x59, 0x7F, 0x43, 255),
        ]
        birch2[9:16] = leaf_grad

        # Row 8: indices 57-64 (trunk) - 8-step trunk gradient
        trunk_grad = [
            (0xFD, 0xFD, 0xFC, 255),
            (0xF8, 0xF8, 0xF7, 255),
            (0xF3, 0xF3, 0xF2, 255),
            (0xEE, 0xEE, 0xED, 255),
            (0xEA, 0xEA, 0xE9, 255),
            (0xE5, 0xE5, 0xE4, 255),
            (0xE0, 0xE0, 0xDF, 255),
            (0xDC, 0xDC, 0xDB, 255),
        ]
        birch2[57:64] = trunk_grad

        # Row 9: indices 65-72 (extra trunk)
        extra_trunk = [
            (0xB0, 0xA4, 0x9D, 255),
            (0xAC, 0xA0, 0x99, 255),
            (200, 200, 200, 255),
            (200, 200, 200, 255),
            (200, 200, 200, 255),
            (200, 200, 200, 255),
            (200, 200, 200, 255),
            (200, 200, 200, 255),
        ]
        birch2[65:72] = extra_trunk

        # Fill other rows with light gray gradients
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8, 9):
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                birch2[i] = (
                    int(180 + 60 * t),
                    int(180 + 60 * t),
                    int(180 + 60 * t),
                    255
                )

        self.register_palette('Birch Variant 2', birch2, map_config={'leaves': list(range(9, 16)), 'trunk': list(range(57, 66))})


        # Palm default: tropical palm greens and browns, row-aligned for MagicaVoxel
        palm_default = [(8, 8, 8, 255)] * 256

        # Row 2: indices 9-16 (leaves) - palm frond green gradient
        leaf_grad = [
            (0x58, 0x76, 0x33, 255),  # index 9: 587633
            (0x55, 0x74, 0x33, 255),  # index 10: 557433
            (0x52, 0x72, 0x33, 255),  # index 11: 527233
            (0x4f, 0x70, 0x33, 255),  # index 12: 4f7033
            (0x4b, 0x6f, 0x32, 255),  # index 13: 4b6f32
            (0x48, 0x6d, 0x32, 255),  # index 14: 486d32
            (0x45, 0x6b, 0x32, 255),  # index 15: 456b32
            (0x42, 0x69, 0x32, 255),  # index 16: 426932
        ]
        palm_default[9:17] = leaf_grad

        # Row 8: indices 57-64 (trunk) - palm trunk brown gradient
        trunk_grad = [
            (0x94, 0x86, 0x71, 255),  # index 57: 948671
            (0x8b, 0x7c, 0x68, 255),  # index 58: 8b7c68
            (0x83, 0x71, 0x5f, 255),  # index 59: 83715f
            (0x7a, 0x68, 0x56, 255),  # index 60: 7a6856
            (0x74, 0x62, 0x50, 255),  # index 61: 746250
            (0x6d, 0x5b, 0x49, 255),  # index 62: 6d5b49
            (0x67, 0x53, 0x43, 255),  # index 63: 675343
            (0x60, 0x4e, 0x3d, 255),  # index 64: 604e3d
        ]
        palm_default[57:65] = trunk_grad

        # Fill other rows with tropical gradients
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row == 2 or row == 8:
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                palm_default[i] = (
                    int(60 + 80 * t),
                    int(100 + 60 * t),
                    int(30 + 50 * t),
                    255
                )

        self.register_palette('Palm Default', palm_default, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

# Palm Lush: deeper, saturated tropical greens with darker trunk
        palm_lush = [(8, 8, 8, 255)] * 256
        leaf_grad = [
            (0x2E, 0x7A, 0x2A, 255),  # index 9
            (0x31, 0x78, 0x2D, 255),  # index 10
            (0x35, 0x76, 0x30, 255),  # index 11
            (0x39, 0x74, 0x33, 255),  # index 12
            (0x3D, 0x71, 0x36, 255),  # index 13
            (0x41, 0x6F, 0x39, 255),  # index 14
            (0x46, 0x6D, 0x3D, 255),  # index 15
            (0x4B, 0x6B, 0x41, 255),  # index 16
        ]
        palm_lush[9:17] = leaf_grad
        trunk_grad = [
            (0x5A, 0x48, 0x36, 255),
            (0x54, 0x44, 0x33, 255),
            (0x4F, 0x40, 0x30, 255),
            (0x4A, 0x3C, 0x2E, 255),
            (0x45, 0x37, 0x2B, 255),
            (0x40, 0x33, 0x29, 255),
            (0x3B, 0x2F, 0x27, 255),
            (0x36, 0x2B, 0x25, 255),
        ]
        palm_lush[57:65] = trunk_grad
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8):
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                palm_lush[i] = (
                    int(30 + 90 * t),
                    int(80 + 90 * t),
                    int(20 + 40 * t),
                    255
                )
        self.register_palette('Palm Lush', palm_lush, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Palm Dusk: cool bluish fronds that read well in low light, slightly muted trunk
        palm_dusk = [(8, 8, 8, 255)] * 256
        leaf_grad = [
            (0x2A, 0x66, 0x46, 255),
            (0x28, 0x62, 0x44, 255),
            (0x26, 0x5E, 0x42, 255),
            (0x24, 0x5A, 0x40, 255),
            (0x22, 0x56, 0x3E, 255),
            (0x20, 0x52, 0x3C, 255),
            (0x1E, 0x4E, 0x3A, 255),
            (0x1C, 0x4A, 0x38, 255),
        ]
        palm_dusk[9:17] = leaf_grad
        trunk_grad = [
            (0x6A, 0x55, 0x4A, 255),
            (0x63, 0x4F, 0x45, 255),
            (0x5C, 0x49, 0x40, 255),
            (0x55, 0x43, 0x3B, 255),
            (0x4E, 0x3E, 0x36, 255),
            (0x47, 0x38, 0x31, 255),
            (0x40, 0x33, 0x2C, 255),
            (0x39, 0x2E, 0x27, 255),
        ]
        palm_dusk[57:65] = trunk_grad
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8):
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                palm_dusk[i] = (
                    int(30 + 50 * t),
                    int(60 + 40 * t),
                    int(30 + 50 * t),
                    255
                )
        self.register_palette('Palm Dusk', palm_dusk, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Palm Sandbar: pale yellow-greens and sandy trunk for beachy palms
        palm_sandbar = [(8, 8, 8, 255)] * 256
        leaf_grad = [
            (0x7A, 0x7A, 0x34, 255),
            (0x78, 0x78, 0x33, 255),
            (0x76, 0x76, 0x32, 255),
            (0x74, 0x74, 0x31, 255),
            (0x72, 0x72, 0x30, 255),
            (0x70, 0x70, 0x2F, 255),
            (0x6E, 0x6E, 0x2E, 255),
            (0x6C, 0x6C, 0x2D, 255),
        ]
        palm_sandbar[9:17] = leaf_grad
        trunk_grad = [
            (0xA8, 0x94, 0x7A, 255),
            (0xA2, 0x8F, 0x76, 255),
            (0x9C, 0x8A, 0x72, 255),
            (0x96, 0x85, 0x6E, 255),
            (0x90, 0x80, 0x6A, 255),
            (0x8A, 0x7B, 0x66, 255),
            (0x84, 0x76, 0x62, 255),
            (0x7E, 0x71, 0x5E, 255),
        ]
        palm_sandbar[57:65] = trunk_grad
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8):
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                palm_sandbar[i] = (
                    int(80 + 60 * t),
                    int(90 + 60 * t),
                    int(20 + 30 * t),
                    255
                )
        self.register_palette('Palm Sandbar', palm_sandbar, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Palm Sunset: warm orange-tinged fronds and richer trunk for evening scenes
        palm_sunset = [(8, 8, 8, 255)] * 256
        leaf_grad = [
            (0x6A, 0x62, 0x2C, 255),
            (0x68, 0x5E, 0x2A, 255),
            (0x66, 0x5A, 0x28, 255),
            (0x64, 0x56, 0x26, 255),
            (0x62, 0x52, 0x24, 255),
            (0x60, 0x4E, 0x22, 255),
            (0x5E, 0x4A, 0x20, 255),
            (0x5C, 0x46, 0x1E, 255),
        ]
        palm_sunset[9:17] = leaf_grad
        trunk_grad = [
            (0x8F, 0x5A, 0x3E, 255),
            (0x88, 0x55, 0x39, 255),
            (0x81, 0x50, 0x34, 255),
            (0x7A, 0x4B, 0x2F, 255),
            (0x73, 0x46, 0x2A, 255),
            (0x6C, 0x41, 0x25, 255),
            (0x65, 0x3C, 0x20, 255),
            (0x5E, 0x37, 0x1B, 255),
        ]
        palm_sunset[57:65] = trunk_grad
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8):
                continue
            for i in range(start, end):
                if i >= 256:
                    break
                t = (i - 1) / 255
                palm_sunset[i] = (
                    int(50 + 120 * t),
                    int(40 + 80 * t),
                    int(15 + 40 * t),
                    255
                )
        self.register_palette('Palm Sunset', palm_sunset, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Kapok family: tropical broadleaf palettes (row-aligned for MagicaVoxel)
        kapok_tropical = [(8, 8, 8, 255)] * 256
        # Row 2: indices 9-16 (leaves) - vibrant mixed green/yellow canopy
        leaf_grad = [
            (0x3A, 0x7A, 0x34, 255),
            (0x44, 0x84, 0x38, 255),
            (0x4E, 0x8E, 0x3C, 255),
            (0x58, 0x98, 0x40, 255),
            (0x6A, 0xA8, 0x4A, 255),
            (0x7C, 0xB8, 0x54, 255),
            (0x8E, 0xC8, 0x5E, 255),
            (0xA0, 0xD8, 0x68, 255),
        ]
        kapok_tropical[9:17] = leaf_grad
        # Row 8: indices 57-64 (trunk) - warm, slightly mottled trunk
        trunk_grad = [
            (0x7A, 0x5A, 0x45, 255),
            (0x76, 0x58, 0x43, 255),
            (0x72, 0x56, 0x41, 255),
            (0x6E, 0x54, 0x3F, 255),
            (0x6A, 0x52, 0x3D, 255),
            (0x66, 0x50, 0x3B, 255),
            (0x62, 0x4E, 0x39, 255),
            (0x5E, 0x4C, 0x37, 255),
        ]
        kapok_tropical[57:65] = trunk_grad
        # Fill other rows with lush mid-greens for body of palette
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8):
                continue
            for i in range(start, end):
                if i >= 256: break
                t = (i - 1) / 255
                kapok_tropical[i] = (
                    int(30 + 120 * t),
                    int(60 + 130 * t) % 256,
                    int(20 + 80 * t),
                    255
                )
        self.register_palette('Kapok Tropical', kapok_tropical, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Kapok Lush: denser, deeper greens + bright upper canopy highlights
        kapok_lush = [(8, 8, 8, 255)] * 256
        leaf_grad = [
            (0x26, 0x62, 0x2A, 255),
            (0x2C, 0x6A, 0x2E, 255),
            (0x32, 0x72, 0x32, 255),
            (0x38, 0x7A, 0x36, 255),
            (0x44, 0x86, 0x3E, 255),
            (0x50, 0x92, 0x46, 255),
            (0x5C, 0x9E, 0x4E, 255),
            (0x68, 0xAA, 0x56, 255),
        ]
        kapok_lush[9:17] = leaf_grad
        trunk_grad = [
            (0x6E, 0x58, 0x45, 255),
            (0x6A, 0x56, 0x43, 255),
            (0x66, 0x54, 0x41, 255),
            (0x62, 0x52, 0x3F, 255),
            (0x5E, 0x50, 0x3D, 255),
            (0x5A, 0x4E, 0x3B, 255),
            (0x56, 0x4C, 0x39, 255),
            (0x52, 0x4A, 0x37, 255),
        ]
        kapok_lush[57:65] = trunk_grad
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8): continue
            for i in range(start, end):
                if i >= 256: break
                t = (i - 1) / 255
                kapok_lush[i] = (
                    int(20 + 80 * t),
                    int(40 + 120 * t) % 256,
                    int(10 + 50 * t),
                    255
                )
        self.register_palette('Kapok Lush', kapok_lush, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Kapok Dusk: warm highlights, yellow/orange top canopy and darker trunk for sunset scenes
        kapok_dusk = [(8, 8, 8, 255)] * 256
        leaf_grad = [
            (0x7A, 0x8A, 0x2A, 255),
            (0x85, 0x92, 0x2E, 255),
            (0x90, 0x9A, 0x32, 255),
            (0x9B, 0xA2, 0x36, 255),
            (0xB0, 0x8C, 0x2A, 255),
            (0xC6, 0x72, 0x22, 255),
            (0xE0, 0x58, 0x18, 255),
            (0xF4, 0x3E, 0x0E, 255),
        ]
        kapok_dusk[9:17] = leaf_grad
        trunk_grad = [
            (0x82, 0x66, 0x4F, 255),
            (0x7E, 0x64, 0x4D, 255),
            (0x7A, 0x62, 0x4B, 255),
            (0x76, 0x60, 0x49, 255),
            (0x72, 0x5E, 0x47, 255),
            (0x6E, 0x5C, 0x45, 255),
            (0x6A, 0x5A, 0x43, 255),
            (0x66, 0x58, 0x41, 255),
        ]
        kapok_dusk[57:65] = trunk_grad
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8): continue
            for i in range(start, end):
                if i >= 256: break
                t = (i - 1) / 255
                kapok_dusk[i] = (
                    int(40 + 100 * t),
                    int(30 + 60 * t),
                    int(10 + 30 * t),
                    255
                )
        self.register_palette('Kapok Dusk', kapok_dusk, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

        # Kapok Rainforest: Deep emerald greens for a dense, humid jungle feel, with subtle golden trunk flecks
        kapok_rainforest = [(8, 8, 8, 255)] * 256
        leaf_grad = [
            (0x1A, 0x4A, 0x1A, 255),  # Deep forest green
            (0x20, 0x52, 0x20, 255),
            (0x26, 0x5A, 0x26, 255),
            (0x2C, 0x62, 0x2C, 255),
            (0x32, 0x6A, 0x32, 255),
            (0x38, 0x72, 0x38, 255),
            (0x3E, 0x7A, 0x3E, 255),
            (0x44, 0x82, 0x44, 255),  # Bright emerald top
        ]
        kapok_rainforest[9:17] = leaf_grad
        trunk_grad = [
            (0x5A, 0x4A, 0x3A, 255),  # Mottled brown with gold flecks
            (0x60, 0x50, 0x40, 255),
            (0x66, 0x56, 0x46, 255),
            (0x6C, 0x5C, 0x4C, 255),
            (0x72, 0x62, 0x52, 255),
            (0x78, 0x68, 0x58, 255),
            (0x7E, 0x6E, 0x5E, 255),
            (0x84, 0x74, 0x64, 255),
        ]
        kapok_rainforest[57:65] = trunk_grad
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8): continue
            for i in range(start, end):
                if i >= 256: break
                t = (i - 1) / 255
                kapok_rainforest[i] = (
                    int(20 + 100 * t),
                    int(40 + 120 * t) % 256,
                    int(15 + 60 * t),
                    255
                )
        self.register_palette('Kapok Rainforest', kapok_rainforest, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

                # Kapok Exotic: Electric pinks and purples for an otherworldly, bioluminescent tropical paradise
        kapok_exotic = [(8, 8, 8, 255)] * 256
        leaf_grad = [
            (0xFF, 0x69, 0xB4, 255),  # Hot pink
            (0xFF, 0x77, 0xC1, 255),
            (0xFF, 0x85, 0xCE, 255),
            (0xFF, 0x93, 0xDB, 255),
            (0xDA, 0x70, 0xD6, 255),  # Orchid purple
            (0xC7, 0x7D, 0xF0, 255),
            (0xB4, 0x8A, 0xFF, 255),
            (0xA1, 0x97, 0xFF, 255),
        ]
        kapok_exotic[9:17] = leaf_grad
        trunk_grad = [
            (0x8B, 0x45, 0x13, 255),  # Saddle brown with purple tint
            (0x91, 0x4B, 0x1A, 255),
            (0x97, 0x51, 0x21, 255),
            (0x9D, 0x57, 0x28, 255),
            (0xA3, 0x5D, 0x2F, 255),
            (0xA9, 0x63, 0x36, 255),
            (0xAF, 0x69, 0x3D, 255),
            (0xB5, 0x6F, 0x44, 255),
        ]
        kapok_exotic[57:65] = trunk_grad
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8): continue
            for i in range(start, end):
                if i >= 256: break
                t = (i - 1) / 255
                kapok_exotic[i] = (
                    int(200 + 55 * t) % 256,
                    int(100 + 100 * t) % 256,
                    int(150 + 100 * t) % 256,
                    255
                )
        self.register_palette('Kapok Exotic', kapok_exotic, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

                # Kapok Neon: Glowing electric blues and cyans, like a cyber-tropical forest under neon lights
        kapok_neon = [(8, 8, 8, 255)] * 256
        leaf_grad = [
            (0x00, 0xFF, 0xFF, 255),  # Cyan
            (0x1A, 0xFF, 0xFF, 255),
            (0x33, 0xFF, 0xFF, 255),
            (0x4D, 0xFF, 0xFF, 255),
            (0x00, 0xBF, 0xFF, 255),  # Deep sky blue
            (0x1A, 0xD1, 0xFF, 255),
            (0x33, 0xE3, 0xFF, 255),
            (0x4D, 0xF5, 0xFF, 255),
        ]
        kapok_neon[9:17] = leaf_grad
        trunk_grad = [
            (0x2F, 0x4F, 0x4F, 255),  # Dark slate gray with blue tint
            (0x35, 0x55, 0x55, 255),
            (0x3B, 0x5B, 0x5B, 255),
            (0x41, 0x61, 0x61, 255),
            (0x47, 0x67, 0x67, 255),
            (0x4D, 0x6D, 0x6D, 255),
            (0x53, 0x73, 0x73, 255),
            (0x59, 0x79, 0x79, 255),
        ]
        kapok_neon[57:65] = trunk_grad
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8): continue
            for i in range(start, end):
                if i >= 256: break
                t = (i - 1) / 255
                kapok_neon[i] = (
                    int(0 + 100 * t),
                    int(100 + 155 * t) % 256,
                    int(200 + 55 * t) % 256,
                    255
                )
        self.register_palette('Kapok Neon', kapok_neon, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

                # Kapok Mystic: Mystical purples and golds, evoking an enchanted forest with glowing auras
        kapok_mystic = [(8, 8, 8, 255)] * 256
        leaf_grad = [
            (0x8A, 0x2B, 0xE2, 255),  # Blue violet
            (0x93, 0x30, 0xE8, 255),
            (0x9C, 0x35, 0xEE, 255),
            (0xA5, 0x3A, 0xF4, 255),
            (0xD4, 0xAF, 0x37, 255),  # Gold
            (0xDA, 0xB5, 0x3D, 255),
            (0xE0, 0xBB, 0x43, 255),
            (0xE6, 0xC1, 0x49, 255),
        ]
        kapok_mystic[9:17] = leaf_grad
        trunk_grad = [
            (0x69, 0x69, 0x69, 255),  # Dim gray with purple undertones
            (0x6F, 0x6F, 0x6F, 255),
            (0x75, 0x75, 0x75, 255),
            (0x7B, 0x7B, 0x7B, 255),
            (0x81, 0x81, 0x81, 255),
            (0x87, 0x87, 0x87, 255),
            (0x8D, 0x8D, 0x8D, 255),
            (0x93, 0x93, 0x93, 255),
        ]
        kapok_mystic[57:65] = trunk_grad
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8): continue
            for i in range(start, end):
                if i >= 256: break
                t = (i - 1) / 255
                kapok_mystic[i] = (
                    int(100 + 100 * t) % 256,
                    int(50 + 100 * t) % 256,
                    int(150 + 100 * t) % 256,
                    255
                )
        self.register_palette('Kapok Mystic', kapok_mystic, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

                # Kapok Aurora: Shimmering greens and blues like aurora borealis over a tropical tree
        kapok_aurora = [(8, 8, 8, 255)] * 256
        leaf_grad = [
            (0x00, 0xFF, 0x7F, 255),  # Spring green
            (0x1A, 0xFF, 0x85, 255),
            (0x33, 0xFF, 0x8B, 255),
            (0x4D, 0xFF, 0x91, 255),
            (0x00, 0x7F, 0xFF, 255),  # Azure
            (0x1A, 0x8B, 0xFF, 255),
            (0x33, 0x97, 0xFF, 255),
            (0x4D, 0xA3, 0xFF, 255),
        ]
        kapok_aurora[9:17] = leaf_grad
        trunk_grad = [
            (0x4B, 0x4B, 0x4B, 255),  # Dark gray with blue-green tint
            (0x51, 0x51, 0x51, 255),
            (0x57, 0x57, 0x57, 255),
            (0x5D, 0x5D, 0x5D, 255),
            (0x63, 0x63, 0x63, 255),
            (0x69, 0x69, 0x69, 255),
            (0x6F, 0x6F, 0x6F, 255),
            (0x75, 0x75, 0x75, 255),
        ]
        kapok_aurora[57:65] = trunk_grad
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8): continue
            for i in range(start, end):
                if i >= 256: break
                t = (i - 1) / 255
                kapok_aurora[i] = (
                    int(0 + 100 * t),
                    int(100 + 155 * t) % 256,
                    int(50 + 150 * t) % 256,
                    255
                )
        self.register_palette('Kapok Aurora', kapok_aurora, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

                # Kapok Inferno: Blazing reds and oranges, like a tree engulfed in volcanic flames
        kapok_inferno = [(8, 8, 8, 255)] * 256
        leaf_grad = [
            (0xFF, 0x45, 0x00, 255),  # Orange red
            (0xFF, 0x4F, 0x0A, 255),
            (0xFF, 0x59, 0x14, 255),
            (0xFF, 0x63, 0x1E, 255),
            (0xFF, 0x69, 0x28, 255),  # Red orange
            (0xFF, 0x73, 0x32, 255),
            (0xFF, 0x7D, 0x3C, 255),
            (0xFF, 0x87, 0x46, 255),
        ]
        kapok_inferno[9:17] = leaf_grad
        trunk_grad = [
            (0x8B, 0x45, 0x13, 255),  # Saddle brown with red glow
            (0x95, 0x4A, 0x16, 255),
            (0x9F, 0x4F, 0x19, 255),
            (0xA9, 0x54, 0x1C, 255),
            (0xB3, 0x59, 0x1F, 255),
            (0xBD, 0x5E, 0x22, 255),
            (0xC7, 0x63, 0x25, 255),
            (0xD1, 0x68, 0x28, 255),
        ]
        kapok_inferno[57:65] = trunk_grad
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8): continue
            for i in range(start, end):
                if i >= 256: break
                t = (i - 1) / 255
                kapok_inferno[i] = (
                    int(200 + 55 * t) % 256,
                    int(50 + 100 * t) % 256,
                    int(0 + 50 * t),
                    255
                )
        self.register_palette('Kapok Inferno', kapok_inferno, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

                # Kapok Crystal: Icy blues and whites, like a tree encased in crystal ice in a tropical setting
        kapok_crystal = [(8, 8, 8, 255)] * 256
        leaf_grad = [
            (0xE0, 0xF6, 0xFF, 255),  # Alice blue
            (0xD6, 0xEC, 0xFF, 255),
            (0xCC, 0xE2, 0xFF, 255),
            (0xC2, 0xD8, 0xFF, 255),
            (0xB8, 0xCE, 0xFF, 255),  # Light blue
            (0xAE, 0xC4, 0xFF, 255),
            (0xA4, 0xBA, 0xFF, 255),
            (0x9A, 0xB0, 0xFF, 255),
        ]
        kapok_crystal[9:17] = leaf_grad
        trunk_grad = [
            (0xF5, 0xF5, 0xF5, 255),  # White smoke
            (0xEB, 0xEB, 0xEB, 255),
            (0xE1, 0xE1, 0xE1, 255),
            (0xD7, 0xD7, 0xD7, 255),
            (0xCD, 0xCD, 0xCD, 255),
            (0xC3, 0xC3, 0xC3, 255),
            (0xB9, 0xB9, 0xB9, 255),
            (0xAF, 0xAF, 0xAF, 255),
        ]
        kapok_crystal[57:65] = trunk_grad
        for row in range(1, 33):
            start = (row - 1) * 8 + 1
            end = start + 8
            if row in (2, 8): continue
            for i in range(start, end):
                if i >= 256: break
                t = (i - 1) / 255
                kapok_crystal[i] = (
                    int(200 + 55 * t) % 256,
                    int(220 + 35 * t) % 256,
                    int(240 + 15 * t) % 256,
                    255
                )
        self.register_palette('Kapok Crystal', kapok_crystal, map_config={'leaves': [9, 16], 'trunk': [57, 64]})

# Singleton convenience instance used by the application
_palette_manager = InternalPaletteManager()


def get_internal_palette(name: Optional[str] = None) -> (List[RGBA], Dict[str, List[int]]):
    """Public helper to fetch an internal palette and its semantic map."""
    return _palette_manager.get_palette(name)
