# PixSorter 
![](examples/Cover_1.png)
**Find the BEST photos in the bunch (offline, on your machine).**

PixSorter helps you:
- **Group near-duplicate photos** (bursts, repeats, same scene/pose)
- **Pick the TOP-K best shots** per group using a quality score (sharpness + composition + straightness, with optional eyes-open preference)
- Export results into clean folders so you can keep your library tidy.

> Built with PySide6 (desktop GUI), runs locally, no cloud upload required.

---

## What it does

### ✅ 1) Groups duplicates (fast → accurate)
PixSorter uses a two-stage approach:
1. **CLIP embeddings** to quickly find candidate neighbors (cosine similarity)
2. **ORB + homography (RANSAC) verification** to confirm “same frame” matches

Then it builds connected components to form duplicate groups.

### ✅ 2) Exports a clean folder structure
Given an output folder, PixSorter writes:

```
Output/
  group_0001/
  group_0002/
  ...
  SINGLES/
  BEST_OF/              (optional)
```

- `group_####` contains duplicates/near-duplicates
- `SINGLES` contains photos not assigned to any group
- `BEST_OF` contains TOP-K picks per group (copy) if enabled

### ✅ 3) Picks BEST-OF images (optional)
For each group, PixSorter scores photos based on:
- **Sharpness** (focus, ROI-based)
- **Composition** (saliency centroid, rule-of-thirds / golden ratio proximity, leading lines)
- **Straightness** (tilt estimation)
- **Eyes-open preference** (optional, via MediaPipe FaceMesh)

It can also enforce **diversity** within TOP-K to avoid selecting multiple near-identical burst variants.

---

## Supported formats
PixSorter looks for images with these extensions:

`.jpg .jpeg .png .webp .tif .tiff .bmp .heic .heif`

> HEIC/HEIF reading is optional (see install extras).

---

## Install

### 1) Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install PixSorter (recommended: with extras)
From the repo root:

```bash
pip install -U pip
pip install -e ".[clip,eyes,heic]"
```

**Extras**
- `clip` → installs torch + open_clip (required for the CLIP embedding stage)
- `eyes` → installs mediapipe (eyes-open preference)
- `heic` → installs pillow-heif (HEIC/HEIF support)

> If you install without `clip`, the app will launch but the pipeline will stop at the CLIP stage and tell you to install extras.

---

## Run

### Option A: GUI (recommended)
```bash
pixsorter
```

### Option B: Module run
```bash
python -m pixsorter
```

---

## How to use (GUI)
1. **Drag & drop** a folder of photos (input)
2. Choose an **output folder**
3. (Optional) tick “Create new output folder inside selected output”
4. Click **Apply recommended** (great defaults to start)
5. Hit **Start**

Tip: the **“?”** buttons explain each parameter + recommended values.

---

## Parameters (quick guide)

- **Mode**: `copy` (safe) or `move` (destructive)
- **Min group size**: minimum images required to form a group folder
- **CLIP model**:  
  - `vit_l_14` (more accurate, heavier)  
  - `vit_b_32` (faster, lighter)
- **Neighbors (K)**: how many candidates to verify per image
- **Candidate similarity**: CLIP similarity threshold before ORB verification
- **ORB features / Match strictness / Min inliers / Verify resize**: controls verification strength vs speed
- **Create BEST_OF**: export TOP-K best per group
- **Diversity**: prevents selecting near-identical burst variants
- **Prefer eyes open**: penalizes closed eyes when faces are detected

---

## Performance tips
- For large folders:
  - Reduce **Neighbors (K)** (e.g., 12–18)
  - Reduce **Verify resize** (e.g., 700–900)
  - Use `vit_b_32` for speed
- If you’re getting false positives:
  - Increase **Min inliers**
  - Lower **Match strictness** (stricter) or raise **Candidate similarity**

---

## Packaging (Windows/macOS)
A PyInstaller spec is included:

```bash
pip install pyinstaller
pyinstaller packaging/PixSorter.spec
```

---

## Troubleshooting

### “CLIP not available…”
Install the CLIP extra:
```bash
pip install -e ".[clip]"
```

### HEIC files not loading
Install the HEIC extra:
```bash
pip install -e ".[heic]"
```

### Eyes-open preference not working
Install the eyes extra:
```bash
pip install -e ".[eyes]"
```

---

## License
See `LICENSE`.

---

## Upcoming Updates
- Persistent cache for CLIP embeddings
- Face-aware “subject sharpness” (better focus scoring)
- Session presets (save/load parameter sets)
- Live ETA + throughput display

source .venv/bin/activate

