# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PKG = SRC / "pixsorter"

datas = []
# bundle icon if present
ico = PKG / "assets" / "app.ico"
if ico.exists():
    datas.append((str(ico), "pixsorter/assets"))

hiddenimports = []

def try_collect(pkg_name: str):
    try:
        from PyInstaller.utils.hooks import collect_all
        d, b, h = collect_all(pkg_name)
        return d, b, h
    except Exception:
        return [], [], []

# Optional heavy deps (won't break spec if missing)
for optional in ["mediapipe", "open_clip", "torch", "torchvision", "pillow_heif"]:
    d, b, h = try_collect(optional)
    datas.extend(d)
    hiddenimports.extend(h)

a = Analysis(
    ["-m", "pixsorter"],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=list(set(hiddenimports)),
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PixSorter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(ico) if ico.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="PixSorter",
)