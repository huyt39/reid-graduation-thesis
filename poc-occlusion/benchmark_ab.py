"""A/B Benchmark: Baseline vs Occlusion-Aware Pipeline.

Runs both approaches on the same video and compares:
1. Embedding consistency per track (cosine sim within same person's embeddings)
2. Cross-track separation (cosine sim between different persons)
3. Visibility score distribution (shows scoring works)
4. Visual comparison of selected vs rejected frames

Usage:
    cd poc-occlusion
    PYTHONPATH=edge python3 benchmark_ab.py

Requires: model_serving running on localhost:8000
"""

import asyncio
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

import importlib

# Import edge modules
sys.path.insert(0, "edge")
import src.detection.yolo as _yolo
import src.scoring.visibility as _vis
import src.scoring.tagging as _tag
YoloModel = _yolo.YoloModel
compute_subscores = _vis.compute_subscores
compute_visibility_score = _vis.compute_visibility_score
compute_overlap_ratio = _vis.compute_overlap_ratio
tag_detection = _tag.tag_detection

# Clear src from modules to avoid conflicts, then import worker
_edge_src_modules = [k for k in sys.modules if k.startswith("src")]
for k in _edge_src_modules:
    del sys.modules[k]
sys.path.remove("edge")
sys.path.insert(0, "worker")

from src.utils.ops import xyxy2xywh
from src.scoring.enhanced_visibility import compute_iou_prev, compute_vel_smooth, compute_v_worker
from src.tracking.byte_tracker import BYTETracker
from src.tracklet.models import TrackletEntry
from src.tracklet.selector import TopKSelector
from src.tracklet.consistency import compute_tracklet_consistency
from src.embedding.aggregator import WeightedEmbeddingAggregator
from src.embedding.client import ModelServiceClient

# ── Config ──────────────────────────────────────────────────────────────
VIDEO_PATH = "data/18156284-hd_1080_1920_25fps.mp4"
MODEL_PATH = "yolo11n.pt"
MAX_FRAMES = 300        # Process this many frames
SKIP_RATE = 2           # Pre-frame skip
MODEL_SERVICE_URL = "http://localhost:8000"
TOP_K = 5
MIN_TRACK_FRAMES = 10   # Only evaluate tracks with >= this many frames


@dataclass
class TrackData:
    """Accumulated data for one tracked person."""
    track_id: int
    bboxes: list          # list of [x1,y1,x2,y2]
    crops: list           # list of np.ndarray
    v_scores: list        # visibility scores (PoC)
    overlap_ratios: list  # overlap ratios
    frame_idxs: list


def run_detection_and_tracking(video_path: str, max_frames: int) -> dict[int, TrackData]:
    """Run YOLO + BYTETracker, collect all crops and scores per track."""
    from types import SimpleNamespace

    yolo = YoloModel(model_path=MODEL_PATH, conf_threshold=0.25, imgsz=1280)
    tracker_args = SimpleNamespace(
        track_high_thresh=0.7, track_low_thresh=0.35,
        match_thresh=0.3, new_track_thresh=0.82,
        track_buffer=30, fuse_score=True,
    )
    tracker = BYTETracker(tracker_args, frame_rate=30)
    prev_bboxes: dict[int, list] = {}
    tracks: dict[int, TrackData] = {}

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    processed = 0

    print(f"[1/4] Running detection + tracking on {video_path}...")

    while True:
        ret, frame = cap.read()
        if not ret or processed >= max_frames:
            break
        frame_idx += 1
        if frame_idx % SKIP_RATE != 0:
            continue
        processed += 1

        h, w = frame.shape[:2]
        detections = yolo.infer(frame)
        if not detections:
            continue

        all_bboxes_raw = [d["bbox"] for d in detections]

        # Prepare for tracker
        bboxes_xywh = []
        scores = []
        classes = []
        det_data = []

        for det in detections:
            bbox = det["bbox"]
            conf = det["confidence"]
            subscores = compute_subscores(bbox, conf, w, h, all_bboxes=all_bboxes_raw)
            v_score = compute_visibility_score(subscores)
            overlap = compute_overlap_ratio(bbox, all_bboxes_raw)

            xywh = xyxy2xywh(np.array(bbox))
            bboxes_xywh.append(xywh)
            scores.append(conf)
            classes.append(0)
            det_data.append({"bbox": bbox, "v_score": v_score, "overlap": overlap})

        bboxes_np = np.array(bboxes_xywh, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)
        classes_np = np.array(classes, dtype=np.float32)

        track_results = tracker.update(scores_np, bboxes_np, classes_np, frame)

        for track in track_results:
            bbox_xyxy = track[:4]
            track_id = int(track[4])

            # Match to nearest detection
            v_score = 0.5
            overlap = 0.0
            min_dist = float("inf")
            for dd in det_data:
                dist = np.linalg.norm(bbox_xyxy - np.array(dd["bbox"]))
                if dist < min_dist:
                    min_dist = dist
                    v_score = dd["v_score"]
                    overlap = dd["overlap"]

            # Enhanced scoring
            prev_list = prev_bboxes.get(track_id, [])
            bbox_prev = prev_list[-1] if prev_list else None
            center_curr = np.array([(bbox_xyxy[0]+bbox_xyxy[2])/2, (bbox_xyxy[1]+bbox_xyxy[3])/2])
            center_prev = None
            center_prev2 = None
            if bbox_prev is not None:
                center_prev = np.array([(bbox_prev[0]+bbox_prev[2])/2, (bbox_prev[1]+bbox_prev[3])/2])
            if len(prev_list) >= 2:
                bp2 = prev_list[-2]
                center_prev2 = np.array([(bp2[0]+bp2[2])/2, (bp2[1]+bp2[3])/2])
            bbox_size = max(bbox_xyxy[2]-bbox_xyxy[0], bbox_xyxy[3]-bbox_xyxy[1])
            iou_s = compute_iou_prev(bbox_xyxy, bbox_prev)
            vel_s = compute_vel_smooth(center_curr, center_prev, center_prev2, bbox_size)
            v_worker = compute_v_worker(v_score, iou_s, vel_s)

            prev_bboxes.setdefault(track_id, []).append(bbox_xyxy.copy())
            if len(prev_bboxes[track_id]) > 3:
                prev_bboxes[track_id] = prev_bboxes[track_id][-3:]

            # Crop
            x1, y1, x2, y2 = map(int, bbox_xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if track_id not in tracks:
                tracks[track_id] = TrackData(track_id, [], [], [], [], [])
            td = tracks[track_id]
            td.bboxes.append(bbox_xyxy.tolist())
            td.crops.append(crop)
            td.v_scores.append(v_worker)
            td.overlap_ratios.append(overlap)
            td.frame_idxs.append(frame_idx)

        if processed % 50 == 0:
            print(f"  Frame {frame_idx}, tracks so far: {len(tracks)}")

    cap.release()
    print(f"  Done: {processed} frames, {len(tracks)} tracks total")
    return tracks


async def extract_embeddings(crops: list[np.ndarray], client: ModelServiceClient) -> list[np.ndarray]:
    """Extract embeddings for a list of crops."""
    embeddings = []
    for crop in crops:
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        try:
            _, result = await client.extract_features(buf.tobytes())
            emb = np.array(result["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb = emb / norm
            embeddings.append(emb)
        except Exception as e:
            print(f"  Warning: embedding extraction failed: {e}")
    return embeddings


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def mean_pairwise_sim(embeddings: list[np.ndarray]) -> float:
    """Mean pairwise cosine similarity."""
    if len(embeddings) < 2:
        return 1.0
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(cosine_sim(embeddings[i], embeddings[j]))
    return sum(sims) / len(sims)


async def benchmark(tracks: dict[int, TrackData]):
    """Compare baseline vs PoC embedding strategies."""
    # Filter tracks with enough frames
    valid_tracks = {tid: td for tid, td in tracks.items() if len(td.crops) >= MIN_TRACK_FRAMES}
    print(f"\n[2/4] Tracks with >= {MIN_TRACK_FRAMES} frames: {len(valid_tracks)}")

    if not valid_tracks:
        print("Not enough tracks to benchmark. Try increasing MAX_FRAMES or lowering MIN_TRACK_FRAMES.")
        return

    selector = TopKSelector(k=TOP_K, min_temporal_gap=3, overlap_lambda=0.3)
    aggregator = WeightedEmbeddingAggregator(gamma=0.5)

    async with ModelServiceClient(base_url=MODEL_SERVICE_URL) as client:
        baseline_results = {}
        poc_results = {}

        print(f"\n[3/4] Extracting embeddings (this may take a while)...")

        for tid, td in valid_tracks.items():
            n = len(td.crops)
            print(f"\n  Track {tid} ({n} frames):")

            # ── Baseline: embed ALL frames, simple average ──
            # Sample evenly to keep it tractable (max 10)
            baseline_indices = np.linspace(0, n - 1, min(n, 10), dtype=int)
            baseline_crops = [td.crops[i] for i in baseline_indices]
            baseline_embs = await extract_embeddings(baseline_crops, client)

            if len(baseline_embs) < 2:
                print(f"    Skipping (not enough embeddings)")
                continue

            baseline_tracklet_emb = np.mean(baseline_embs, axis=0)
            baseline_tracklet_emb = baseline_tracklet_emb / (np.linalg.norm(baseline_tracklet_emb) + 1e-8)
            baseline_consistency = mean_pairwise_sim(baseline_embs)

            # ── PoC: visibility-scored top-K, weighted average ──
            entries = [
                TrackletEntry(
                    frame_idx=td.frame_idxs[i],
                    crop=td.crops[i],
                    v_score=td.v_scores[i],
                    bbox_xyxy=td.bboxes[i],
                    timestamp_ns=0,
                    overlap_ratio=td.overlap_ratios[i],
                )
                for i in range(n)
            ]
            selected = selector.select(entries)
            poc_crops = [e.crop for e in selected]
            poc_v_scores = [e.v_score for e in selected]
            poc_overlaps = [e.overlap_ratio for e in selected]
            poc_embs = await extract_embeddings(poc_crops, client)

            if len(poc_embs) < 2:
                print(f"    Skipping PoC (not enough embeddings)")
                continue

            poc_tracklet_emb = aggregator.aggregate(poc_embs, poc_v_scores, poc_overlaps)
            poc_consistency = mean_pairwise_sim(poc_embs)

            # Tracklet consistency features
            tc = compute_tracklet_consistency(entries)

            baseline_results[tid] = {
                "consistency": baseline_consistency,
                "tracklet_emb": baseline_tracklet_emb,
                "n_frames": len(baseline_embs),
            }
            poc_results[tid] = {
                "consistency": poc_consistency,
                "tracklet_emb": poc_tracklet_emb,
                "n_frames": len(poc_embs),
                "v_scores": poc_v_scores,
                "tracklet_consistency": tc,
            }

            v_avg = np.mean(td.v_scores)
            v_selected_avg = np.mean(poc_v_scores)

            print(f"    Baseline: consistency={baseline_consistency:.4f} (from {len(baseline_embs)} frames)")
            print(f"    PoC:      consistency={poc_consistency:.4f} (from {len(poc_embs)} top-K frames)")
            print(f"    v_score: all_avg={v_avg:.3f} → selected_avg={v_selected_avg:.3f} (Δ={v_selected_avg-v_avg:+.3f})")
            print(f"    Tracklet: size_stab={tc.bbox_size_stability:.2f} pos_stab={tc.position_stability:.2f} streak={tc.good_frame_streak}")

        # ── Summary ──
        if not baseline_results:
            print("\nNo valid tracks for comparison.")
            return

        print("\n" + "=" * 70)
        print("[4/4] SUMMARY")
        print("=" * 70)

        b_consistencies = [r["consistency"] for r in baseline_results.values()]
        p_consistencies = [r["consistency"] for r in poc_results.values()]

        print(f"\nEmbedding Consistency (higher = same person's embeddings are more similar):")
        print(f"  Baseline:  mean={np.mean(b_consistencies):.4f}  (min={np.min(b_consistencies):.4f}, max={np.max(b_consistencies):.4f})")
        print(f"  PoC:       mean={np.mean(p_consistencies):.4f}  (min={np.min(p_consistencies):.4f}, max={np.max(p_consistencies):.4f})")

        improvement = np.mean(p_consistencies) - np.mean(b_consistencies)
        print(f"  Improvement: {improvement:+.4f} ({'PoC wins' if improvement > 0 else 'Baseline wins'})")

        # Cross-track separation
        tids = list(baseline_results.keys())
        if len(tids) >= 2:
            b_cross = []
            p_cross = []
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    b_cross.append(cosine_sim(
                        baseline_results[tids[i]]["tracklet_emb"],
                        baseline_results[tids[j]]["tracklet_emb"],
                    ))
                    p_cross.append(cosine_sim(
                        poc_results[tids[i]]["tracklet_emb"],
                        poc_results[tids[j]]["tracklet_emb"],
                    ))

            print(f"\nCross-Track Separation (lower = different persons are more separable):")
            print(f"  Baseline:  mean={np.mean(b_cross):.4f}")
            print(f"  PoC:       mean={np.mean(p_cross):.4f}")
            sep_improvement = np.mean(b_cross) - np.mean(p_cross)
            print(f"  Improvement: {sep_improvement:+.4f} ({'PoC wins' if sep_improvement > 0 else 'Baseline wins'})")

        # Score distribution
        print(f"\nVisibility Score Distribution:")
        all_v = []
        for td in valid_tracks.values():
            all_v.extend(td.v_scores)
        print(f"  All frames:  mean={np.mean(all_v):.3f} std={np.std(all_v):.3f}")
        good = sum(1 for v in all_v if v >= 0.7)
        mid = sum(1 for v in all_v if 0.4 <= v < 0.7)
        bad = sum(1 for v in all_v if v < 0.4)
        total = len(all_v)
        print(f"  Distribution: good={good}({100*good/total:.0f}%) mid={mid}({100*mid/total:.0f}%) bad={bad}({100*bad/total:.0f}%)")

        print("\n" + "=" * 70)
        print("CONCLUSION:")
        if improvement > 0.005:
            print("  ✓ PoC produces MORE CONSISTENT embeddings per person")
            print("    → Occlusion-aware scoring + top-K selection works!")
        elif improvement > -0.005:
            print("  ~ Results are similar (video may not have enough occlusion)")
            print("    → Try a video with more crowded scenes / person overlap")
        else:
            print("  ✗ Baseline performs better on this video")
            print("    → Check scoring weights or try different video")
        print("=" * 70)


def main():
    tracks = run_detection_and_tracking(VIDEO_PATH, MAX_FRAMES)
    asyncio.run(benchmark(tracks))


if __name__ == "__main__":
    main()
