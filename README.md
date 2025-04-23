# 🌲 treegen-pinegen

![treegen-pinegen logo](https://cdn.nostrcheck.me/46025249f65d47dddb0f17d93eb8b0a32d97fe3189c6684bbd33136a0a7e0424/ee9d2b2b47e1bb1656f784e78fcf0db54449afc8b3ac8969c54ca9c3cbdf4afa.webp)

Procedural Voxel Tree + Pine Tree Generator for MagicaVoxel

Generate beautiful, customizable .vox trees using palettes, sliders, and pure Python magic.

Built with Python, Tkinter, and NumPy — no external 3D tools required.

## ✨ Features

- 🌳 Treegen – Oak-style branching tree generator
- 🌲 Pinegen – Pine tree generator with cone-shaped leaf clusters
- 🎨 Custom Palettes – Use .png palettes for different tree types
- 🧩 Tabbed GUI – Switch between tree and pine generation in one app
- 🎛️ Sliders for Everything – Size, twist, branch density, leafiness, and more
- 🪟 Preview Window - Visualize your output in real-time before you generate
- 💾 .VOX Export – Compatible with MagicaVoxel
- 📁 Organized Output – Saves to output/tree/ and output/pine/

## 🚀 How to Run

1. Install dependencies
```bash
pip install pillow numpy
```

2. Run the app
```bash
python treegen-pinegen.py
```

## Downloads

You can also find the pre-compiled .exe under [Releases](https://github.com/NGNT/treegen-pinegen/releases) to get right in.

## 🛠️ Build to .exe (Optional)

You can compile it into a standalone executable using PyInstaller:

```bash
pyinstaller --onefile --windowed --icon=treegen_icon.ico ^
  --add-data "treegen_brand.png;." ^
  --add-data "pinegen_brand.png;." ^
  --add-data "palettes;palettes" ^
  treegen-pinegen.py
```

> 💡 On macOS/Linux, replace `;` with `:` in --add-data paths.

## 🖼️ Palettes

Each palette is a 256x1 PNG image with indexed colors.

- Tree palettes: `palettes/tree/`
- Pine palettes: `palettes/pine/`
- Make sure to add new palettes to the appropriate internal dictionary in the script.

## 👤 Credits

Created by NGNT  
With GUI and architecture support from Cursor and ChatGPT 🤖  
Inspired by nature.

## 📜 License

MIT — Free to use, remix, and plant digital forests 🌳🌲

---