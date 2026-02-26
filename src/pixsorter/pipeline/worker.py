# src/pixsorter/pipeline/worker.py
from __future__ import annotations

import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal
from sklearn.neighbors import NearestNeighbors

from ..config.defaults import SortConfig
from ..infra.resources import is_image
from ..ml.clip_embedder import ClipEmbedder
from ..vision.grouping import connected_components
from ..vision.io import load_rgb, resize_max_side, rgb_to_gray
from ..vision.motion_blur import assess_motion_blur
from ..vision.orb import orb_verify_same_frame
from ..vision.scoring import best_of_score
from .planner import ProgressPlan


class SortWorker(QObject):
    progress = Signal(int)        # 0..100
    stage = Signal(str)           # stage text
    log = Signal(str)             # log lines
    finished = Signal(bool, str)  # success, message

    def __init__(self, cfg: SortConfig):
        super().__init__()
        self.cfg = cfg
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _emit(self, msg: str):
        self.log.emit(msg)

    def _check_cancel(self) -> bool:
        if self._cancel:
            self.finished.emit(False, "Cancelled.")
            return True
        return False

    def run(self):
        plan = ProgressPlan()

        try:
            t0 = time.time()

            in_dir = Path(self.cfg.input_dir).expanduser().resolve()
            out_dir = Path(self.cfg.output_dir).expanduser().resolve()
            if self.cfg.output_create_new:
                out_dir = out_dir / (self.cfg.output_new_name or "SortedOutput")

            out_dir.mkdir(parents=True, exist_ok=True)

            files = [p for p in in_dir.rglob("*") if p.is_file() and is_image(p)]
            if not files:
                self.finished.emit(False, "No images found in input folder.")
                return

            self.stage.emit(f"Found {len(files)} images")
            self._emit(f"Input: {in_dir}")
            self._emit(f"Output: {out_dir}")
            self.progress.emit(1)

            # -------------------------
            # 1) CLIP embeddings
            # -------------------------
            self.stage.emit("Embedding with CLIP (candidate search)…")
            try:
                embedder = ClipEmbedder(clip_model_key=self.cfg.clip_model)
            except Exception as e:
                self.finished.emit(
                    False,
                    f"CLIP not available. Install extras: pip install \"pixsorter[clip]\". ({e})",
                )
                return

            embs: list[np.ndarray] = []
            kept_paths: list[Path] = []

            emit_every = 10  # reduce UI chatter
            for i, p in enumerate(files):
                if self._check_cancel():
                    return

                try:
                    rgb = load_rgb(p)
                    rgb_small = resize_max_side(rgb, 768)
                    emb = embedder.embed_rgb(rgb_small)
                    embs.append(emb)
                    kept_paths.append(p)
                except Exception as e:
                    self._emit(f"[WARN] CLIP embed fail: {p.name} :: {e}")

                if (i + 1) % emit_every == 0 or (i + 1) == len(files):
                    self.progress.emit(plan.embed_pct(i + 1, len(files)))

            if len(kept_paths) < 2:
                self.finished.emit(False, "Not enough readable images after embedding.")
                return

            E = np.vstack(embs)
            nn = NearestNeighbors(
                n_neighbors=min(self.cfg.knn_k + 1, len(kept_paths)),
                metric="cosine",
            )
            nn.fit(E)

            # -------------------------
            # 2) Verify candidates (ORB + homography)
            # -------------------------
            self.stage.emit("Verifying near-duplicates (ORB + homography)…")
            edges: dict[int, list[int]] = defaultdict(list)

            emit_every = 5
            for i in range(len(kept_paths)):
                if self._check_cancel():
                    return

                dists, idxs = nn.kneighbors(E[i : i + 1], return_distance=True)
                for dist, j in zip(dists[0][1:], idxs[0][1:]):
                    sim = 1.0 - float(dist)
                    if sim < self.cfg.clip_sim_thresh:
                        continue
                    if j <= i:
                        continue

                    try:
                        rgb1 = resize_max_side(load_rgb(kept_paths[i]), self.cfg.max_side_verify)
                        rgb2 = resize_max_side(load_rgb(kept_paths[j]), self.cfg.max_side_verify)
                        g1 = rgb_to_gray(rgb1)
                        g2 = rgb_to_gray(rgb2)

                        same, _, _ = orb_verify_same_frame(
                            g1,
                            g2,
                            orb_nfeatures=self.cfg.orb_nfeatures,
                            match_ratio=self.cfg.orb_match_thresh,
                            min_inliers=self.cfg.min_inliers,
                        )
                        if same:
                            edges[i].append(j)
                            edges[j].append(i)
                    except Exception:
                        # keep going; a single bad pair shouldn't stop the run
                        continue

                if (i + 1) % emit_every == 0 or (i + 1) == len(kept_paths):
                    self.progress.emit(plan.verify_pct(i + 1, len(kept_paths)))

            # -------------------------
            # 3) Connected components + write output
            # -------------------------
            self.stage.emit("Creating groups & writing folders…")
            self.progress.emit(ProgressPlan.WRITE.start)

            comps = connected_components(edges, len(kept_paths))
            groups = [g for g in comps if len(g) >= self.cfg.min_group]
            groups = sorted(groups, key=len, reverse=True)

            used = {idx for g in groups for idx in g}
            singles = [kept_paths[i] for i in range(len(kept_paths)) if i not in used]

            op = shutil.copy2 if self.cfg.mode == "copy" else shutil.move

            singles_dir = out_dir / "SINGLES"
            singles_dir.mkdir(exist_ok=True)

            group_dirs: list[Path] = []
            for k, g in enumerate(groups, start=1):
                if self._check_cancel():
                    return

                gd = out_dir / f"group_{k:04d}"
                gd.mkdir(exist_ok=True)
                group_dirs.append(gd)

                for idx in g:
                    if self._check_cancel():
                        return
                    p = kept_paths[idx]
                    dest = gd / p.name
                    if dest.exists():
                        dest = gd / f"{p.stem}__DUP{p.suffix}"
                    op(str(p), str(dest))

            for p in singles:
                if self._check_cancel():
                    return
                dest = singles_dir / p.name
                if dest.exists():
                    dest = singles_dir / f"{p.stem}__DUP{p.suffix}"
                op(str(p), str(dest))

            self._emit(f"Groups created: {len(group_dirs)} (min size {self.cfg.min_group})")
            self._emit(f"Singles: {len(singles)}")
            self.progress.emit(plan.write_pct())

            # -------------------------
            # 4) Best-of TOP-K
            # -------------------------
            if self.cfg.best_of and group_dirs:
                self.stage.emit("Selecting TOP-K best photos per group…")

                best_dir = out_dir / "BEST_OF"
                best_dir.mkdir(exist_ok=True)

                analyzer = None
                if self.cfg.use_eyes:
                    try:
                        from ..ml.eyes import FaceEyeAnalyzer

                        analyzer = FaceEyeAnalyzer()
                    except Exception as e:
                        self._emit(
                            f"[WARN] Eyes-open not available (install extras: pixsorter[eyes]). ({e})"
                        )
                        analyzer = None

                # Blur gate thresholds (0..1 where higher = blurrier)
                CLEAN_MAX = 0.18       # very strict: preferred candidates
                ACCEPT_MAX = 0.32      # still acceptable; below this we try to fill TOP-K

                for gi, gd in enumerate(group_dirs):
                    if self._check_cancel():
                        return

                    imgs = [p for p in gd.iterdir() if p.is_file() and is_image(p)]
                    if not imgs:
                        self.progress.emit(plan.best_pct(gi + 1, len(group_dirs)))
                        continue

                    # (score, path, tag, blur_amount)
                    scored: list[tuple[float, Path, str, float]] = []
                    for p in imgs:
                        if self._check_cancel():
                            return
                        try:
                            rgb = load_rgb(p)
                            rgb = resize_max_side(rgb, 1400)

                            face_boxes: list[tuple[int, int, int, int]] = []
                            eyes_frac = 1.0
                            if analyzer is not None:
                                fb, ef = analyzer.analyze(rgb)
                                face_boxes = fb
                                eyes_frac = ef

                            mb = assess_motion_blur(rgb) or {}
                            tag = str(mb.get("label", "") or "")
                            blur = float(mb.get("blur_amount", 0.0) or 0.0)

                            s = best_of_score(
                                rgb=rgb,
                                face_bboxes=face_boxes,
                                eyes_open_frac=eyes_frac,
                                eyes_weight=self.cfg.eyes_weight,
                            )

                            # Soft penalty for any blur (prevents “slightly blurred but good composition” from winning)
                            s = max(0.0, s - 0.35 * blur)

                            # Extra penalties by motion-blur label
                            if tag == "MB_FAULTY":
                                s = max(0.0, s - 0.35 * blur - 0.10)
                            elif tag == "MB_INTENTIONAL":
                                s = max(0.0, s - 0.10 * blur)

                            scored.append((s, p, tag, blur))
                        except Exception:
                            continue

                    if not scored:
                        self.progress.emit(plan.best_pct(gi + 1, len(group_dirs)))
                        continue

                    scored.sort(reverse=True, key=lambda x: x[0])

                    chosen: list[Path] = []
                    tag_map: dict[Path, str] = {}

                    def passes_diversity(cand: Path) -> bool:
                        if not self.cfg.diverse or not chosen:
                            return True
                        try:
                            rgb_c = resize_max_side(load_rgb(cand), self.cfg.max_side_verify)
                            g_c = rgb_to_gray(rgb_c)

                            for prev in chosen:
                                rgb_p = resize_max_side(load_rgb(prev), self.cfg.max_side_verify)
                                g_p = rgb_to_gray(rgb_p)
                                same, _, _ = orb_verify_same_frame(
                                    g_c,
                                    g_p,
                                    orb_nfeatures=self.cfg.orb_nfeatures,
                                    match_ratio=self.cfg.orb_match_thresh,
                                    min_inliers=self.cfg.diverse_inliers,
                                )
                                if same:
                                    return False
                            return True
                        except Exception:
                            # if diversity check fails, don't block selection
                            return True

                    # Bucket selection order: clean -> acceptable -> intentional -> other non-faulty -> faulty
                    buckets: list[list[tuple[float, Path, str, float]]] = []

                    buckets.append([t for t in scored if t[2] != "MB_FAULTY" and t[3] <= CLEAN_MAX])
                    buckets.append(
                        [t for t in scored if t[2] != "MB_FAULTY" and CLEAN_MAX < t[3] <= ACCEPT_MAX]
                    )
                    buckets.append([t for t in scored if t[2] == "MB_INTENTIONAL"])
                    buckets.append([t for t in scored if t[2] not in ("MB_FAULTY", "MB_INTENTIONAL")])
                    buckets.append([t for t in scored if t[2] == "MB_FAULTY"])

                    seen: set[Path] = set()
                    used_faulty = False

                    for b in buckets:
                        for score, cand, tag, blur in b:
                            if self._check_cancel():
                                return
                            if len(chosen) >= self.cfg.top_k:
                                break
                            if cand in seen:
                                continue
                            seen.add(cand)

                            if not passes_diversity(cand):
                                continue

                            chosen.append(cand)
                            tag_map[cand] = tag
                            if tag == "MB_FAULTY":
                                used_faulty = True

                        if len(chosen) >= self.cfg.top_k:
                            break

                    if used_faulty:
                        self._emit(
                            f"[WARN] {gd.name}: Not enough low-blur images; BEST_OF includes motion-blur shots."
                        )

                    group_name = gd.name
                    for rank, p in enumerate(chosen, start=1):
                        tag = tag_map.get(p, "")
                        tag_part = f"__{tag}" if tag else ""
                        dest = best_dir / f"{group_name}__top{rank:02d}{tag_part}__{p.name}"
                        if dest.exists():
                            dest = best_dir / f"{group_name}__top{rank:02d}__{p.stem}__DUP{p.suffix}"
                        shutil.copy2(str(p), str(dest))

                    self.progress.emit(plan.best_pct(gi + 1, len(group_dirs)))

            self.progress.emit(100)
            dt = time.time() - t0
            self.stage.emit("Done")
            self._emit(f"Completed in {dt:.1f}s")
            self.finished.emit(True, f"Done. Output: {out_dir}")

        except Exception:
            self._emit("ERROR:\n" + traceback.format_exc())
            self.finished.emit(False, "Unexpected error. See log for details.")