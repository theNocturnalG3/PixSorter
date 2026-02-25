from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets_src"
DST = ROOT / "src" / "pixsorter" / "assets"

def main():
    DST.mkdir(parents=True, exist_ok=True)
    ico = SRC / "app.ico"
    if not ico.exists():
        raise FileNotFoundError(f"Missing {ico}. Put your icon there first.")
    shutil.copy2(ico, DST / "app.ico")
    print("✅ Synced assets:", DST / "app.ico")

if __name__ == "__main__":
    main()