# From repo root:
#   python scripts/sync_assets.py
#   .\scripts\build_onedir.ps1

python scripts/sync_assets.py
pyinstaller --clean -y packaging\PixSorter.spec