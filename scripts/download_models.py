from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "src" / "pixsorter" / "assets" / "models"
DST.mkdir(parents=True, exist_ok=True)

# Official model hosting used in many examples
URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
OUT = DST / "face_landmarker.task"

def main():
    if OUT.exists() and OUT.stat().st_size > 1_000_000:
        print("✅ Model already exists:", OUT)
        return
    print("Downloading:", URL)
    urllib.request.urlretrieve(URL, OUT)
    print("✅ Saved:", OUT, f"({OUT.stat().st_size} bytes)")

if __name__ == "__main__":
    main()