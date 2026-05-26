# ReID ID-splitting fix — Baseline notes

Snapshot các thay đổi cho nhiệm vụ giảm ID-splitting trong pipeline ReID. Validated 6/6 person reid đúng trên 2 video độc lập (device2, device3).

## Vấn đề

Pipeline ReID lúc đầu bị **ID-splitting**: một người tách thành nhiều `person_id` khi điều kiện thay đổi (occlusion, scale, ánh sáng, góc nhìn). Baseline trước fix: chỉ 4/6 unique persons được reid đúng, với 1 person nhận 50%+ tracklets (over-merge từ người khác).

## Root causes phát hiện

1. **Embedding model**: pipeline đang dùng OSNet global feature (kém robust occlusion); LMBN multi-branch (kiến trúc designed cho occlusion) đã có code nhưng disable mặc định. Nhánh LMBN trong `model_registry.py` còn thiếu L2-normalize ở output (3 hàm extract) — vector chưa unit-norm gây drift ở EMA/heuristics downstream.

2. **Logic flaw — spatial-first override embedding**: nhiều path trong `reid_worker` cho phép merge/match bypass embedding similarity check khi có spatial signal:
   - `_maybe_accept_occlusion_provisional_match` (workers/main.py:810): `short_reentry=True` hạ threshold xuống 0.58 — bypass main 0.60 threshold.
   - `_persons_have_soft_split_transition` (workers/main.py:3225): `duplicate_box` reason trigger qua 2 path OR (IoU OR center_ratio), với **singleton bypass score check**.
   - `soft_split_can_override_cooccurrence/attributes` (workers/main.py:3129/3156): 5 spatial reasons bypass cooccurrence + attribute guard **bất kể sim**. Comment line 3103-3106 ghi rõ logic chặt chẽ nhưng code dưới vi phạm.

3. **Bbox quality**: crop có person-person overlap ≥ 0.5 thường chứa 2 người → embedding lai → contaminate downstream.

4. **Untracked cluster gate quá khắt khe**: cluster với entries=5, v≥0.85, consistency 0.83-0.87 (gần threshold 0.88) không pass fast tier, không grow lên 6 entries để pass slow tier → người ở xa/occluded ngắn hạn không bao giờ được reid.

5. **Cross-scale embedding shift**: LMBN finetuned cho same-person sim 0.55-0.65 khi crop nhỏ (xa) vs crop lớn (gần) — quá thấp so với threshold 0.75. Không thể fix bằng 1 threshold cho cross-video (video same-scale cross-person sim 0.80-0.89, video cross-scale same-person sim 0.55-0.65 — 2 distributions overlap).

## Các thay đổi áp dụng

### Phase A — Bật LMBN + fix L2-normalize

- `src/inference_engine/src/services/model_registry.py`: thêm L2-normalize cho 3 nhánh LMBN (`extract_embedding`, `extract_embedding_batch`, `extract_embedding_from_tensors`).
- Default `WORKER_EMBEDDING_MODEL=lmbn` (config.py default thay vì env override).
- LMBN weights: `lmbn_n_finetuned.pth` (đã có sẵn ở `src/inference_engine/src/assets/models/lmbn/`). Env `INFERENCE_LMBN_WEIGHTS` trỏ đến file này.

### Phase C — Tune env chặn spatial-bypass-embedding

Các default trong `src/reid_worker/src/core/config.py` đã update theo baseline mới:

| Param | Default cũ | Default mới | Chặn path |
|---|---|---|---|
| `occlusion_provisional_match_threshold` | 0.60 | **0.75** | base provisional path |
| `duplicate_merge_soft_split_duplicate_iou_threshold` | 0.45 | **0.75** | duplicate_box IoU path |
| `duplicate_merge_soft_split_max_center_distance_ratio` | 0.35 | **0.15** | duplicate_box center_ratio path |
| `duplicate_merge_soft_split_duplicate_box_multitrack_min_score` | 0.58 | **0.78** | multitrack bypass |
| `duplicate_merge_overlap_spatial_duplicate_min_score` | 0.58 | **0.78** | overlap_spatial path |

### Phase D — Bbox quality + untracked cluster

- `src/edge/src/core/config.py`: thêm field `hard_drop_overlap_ratio: float = 0.5`.
- `src/edge/src/workers/main.py:349`: drop detection có `overlap_ratio >= hard_drop_overlap_ratio` trước khi gửi Kafka.
- `untracked_cluster_promote_fast_min_overall_consistency`: 0.88 → **0.83** (rescue cluster v=0.99 cons=0.84 không pass tier cũ).

### Phase E — Cross-scale rescue (spatial-continuity-aware)

| Param | Default cũ | Default mới |
|---|---|---|
| `occlusion_provisional_reentry_min_similarity` | 0.58 | **0.55** |
| `occlusion_provisional_reentry_max_gap_frames` | 180 | **120** |

Logic: tracklet với sim 0.55-0.75 chỉ accept nếu đồng thời (a) frame_gap với last person observation ≤ 120 (4s @ 30fps), (b) center_distance_ratio ≤ 2.0. Cho phép same-person cross-scale match mà không bị 2 người khác đứng cùng vị trí mạo nhận.

### Observability (zero behavior change)

`src/reid_worker/src/workers/main.py`: thêm 4 log point rejection để diagnose:
- `occlusion_provisional_rejected_score` / `_margin` / `_no_occlusion_signal`
- `untracked_cluster_rejected`

## Validation

| Video | Persons.count | Ground truth | Notes |
|---|---|---|---|
| device2 | 6 | 6 | sạch, không lẫn ảnh giữa person, tracklet distribution đều |
| device3 | 6 | 6 | sau Phase E fix cross-scale + relax cluster gate |

## Files thay đổi (commit này)

```
src/edge/src/core/config.py            (+hard_drop_overlap_ratio)
src/edge/src/workers/main.py           (+overlap hard drop)
src/inference_engine/src/services/model_registry.py  (+L2-norm LMBN)
src/reid_worker/src/core/config.py     (defaults baseline mới)
src/reid_worker/src/workers/main.py    (+4 log rejection)
scripts/drop_qdrant_persons.sh         (helper — drop Qdrant collection + reset Redis seq)
REID_IDSPLIT_FIX_PLAN.md               (plan file)
REID_IDSPLIT_BASELINE.md               (file này)
```

## Rollback

Mọi tune đều configurable qua env. Rollback bằng cách set env (không cần đổi code):

```bash
# Quay về OSNet
WORKER_EMBEDDING_MODEL=osnet

# Rollback occlusion provisional threshold
WORKER_OCCLUSION_PROVISIONAL_MATCH_THRESHOLD=0.60
WORKER_OCCLUSION_PROVISIONAL_REENTRY_MIN_SIMILARITY=0.58
WORKER_OCCLUSION_PROVISIONAL_REENTRY_MAX_GAP_FRAMES=180

# Rollback duplicate_merge guards
WORKER_DUPLICATE_MERGE_SOFT_SPLIT_DUPLICATE_IOU_THRESHOLD=0.45
WORKER_DUPLICATE_MERGE_SOFT_SPLIT_MAX_CENTER_DISTANCE_RATIO=0.35

# Disable bbox hard drop
EDGE_HARD_DROP_OVERLAP_RATIO=0.99
```

## Future work — Phase B (chưa làm, deferred)

Audit tổng quan logic `ReIDMatcher` (28 threshold/flag) — wire `_record_decision` ra MongoDB, build audit script qua ≥3 video, simplify dead code (good_streak_*, update_sim_threshold), align defaults matcher ↔ Settings (9 params lệch), đơn giản hoá 3 magic constants của `current_identity_switch_*`. Xem `REID_IDSPLIT_FIX_PLAN.md` Phase B.

Hướng cải thiện embedding bền vững hơn tune threshold:
- Fine-tune LMBN với multi-scale augmentation (`scripts/lmbn_finetune_colab.py`) để giảm cross-scale shift.
- Multi-scale gallery (bin embedding theo scale) — code change lớn, để sau khi Phase B ổn.
