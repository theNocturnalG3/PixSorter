# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import os

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# PyInstaller sets SPECPATH to the directory containing the spec file.
SPECDIR = Path(globals().get("SPECPATH", os.getcwd())).resolve()

# Your spec lives in ./packaging, so project root is one level up.
ROOT = SPECDIR.parent if SPECDIR.name.lower() == "packaging" else SPECDIR

SRC = ROOT / "src"
PKG = SRC / "pixsorter"
ENTRY = PKG / "__main__.py"

datas = []
ico = PKG / "assets" / "app.ico"
if ico.exists():
    datas.append((str(ico), "pixsorter/assets"))

hiddenimports = []

# Optional heavy deps (won't crash if missing)
for name in ["mediapipe", "open_clip", "torch", "torchvision", "pillow_heif"]:
    try:
        d, b, h = collect_all(name)
        datas += d
        hiddenimports += h
    except Exception:
        pass

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=list(set(hiddenimports)),
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
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

model = PKG / "assets" / "models" / "face_landmarker.task"
if model.exists():
    datas.append((str(model), "pixsorter/assets/models"))
    
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="PixSorter",
)