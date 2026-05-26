# Plan: Giảm ID-splitting cho pipeline ReID (LMBN + audit logic)

## Context

Pipeline ReID hiện đang gặp **ID-splitting**: một người bị tách thành nhiều `person_id` khi điều kiện thay đổi (occlusion, ánh sáng, tư thế, góc nhìn) — đi ngược tinh thần ReID-trong-occlusion của tài liệu `ReID-Pipeline.pdf`.

Sau khi đối chiếu **6 bước** trong PDF với codebase, kết luận:

- **Logic implementation bám sát spec rất tốt**, thậm chí vượt spec ở nhiều safety nets ngoài PDF (find_duplicate_candidate / current_identity_maintained / ambiguous_rejected / FPS gallery diversity / outlier-aware aggregator).
- **Hai nguồn nghi vấn lớn nhất** gây ID-splitting:
  1. **Model**: backbone LMBN (multi-branch, kiến trúc designed cho occlusion) **bị tắt mặc định** — pipeline chỉ chạy OSNet global feature. File pretrain `lmbn_n_cuhk03_d.pth` đã có sẵn nhưng chưa wire. Đồng thời nhánh LMBN trong `model_registry.py` **thiếu L2-normalize** ở output, dù được wire vẫn drift.
  2. **Logic**: `ReIDMatcher.__init__` có **28 threshold/cờ**, trong đó nhiều cờ ngoài spec PDF (eager_soft_match / soft_match_at_promotion / capped_soft_match / current_identity_switch_*) được thêm qua nhiều vòng iterate — rủi ro đã **overfit cho video test cũ**. Default giữa matcher constructor và `Settings` còn lệch ở 9 tham số. Hai cờ `good_streak_*` không được tham chiếu trong logic (dead).

Plan này không tune cho 1 video cụ thể. Mọi threshold đổi đều phải dựa trên **audit log decisions aggregate qua ≥3 video** (xem Phase B).

## Compliance summary (đối chiếu PDF)

| Bước PDF | Status | Note ngắn |
|---|---|---|
| **Edge - visibility scoring (v)** | OK | cut_off / area_ratio / aspect / det_conf / person-person overlap đầy đủ; weights hợp lý |
| **Edge - tag + post-frame-skip** | OK | good=½, mid=⅓, bad=⅕, drop_floor=0.15 |
| **Worker B1 - tracking + v_worker** | OK | ByteTrack + IoU_prev + vel_smooth |
| **Worker B2 - tracklet buffer + consistency** | PARTIAL | bbox_size/position_stability + good_streak có; nhưng good_streak không vào overall (riêng metric gửi matcher) |
| **Worker B3 - top-K + diversity** | OK | selection_score = v - λ·overlap, temporal_gap, readiness gate đầy đủ |
| **Worker B4 - GeM + weighted + L2** | PARTIAL | GeM OK ở OSNet; **LMBN branch thiếu L2-norm** ở 3 hàm extract |
| **Worker B5 - matching + delayed ID + momentum** | OK + DIVERGES | Delayed-ID gate 4 điều kiện ✅. "Momentum" là design choice **append-to-gallery + FPS prune** (cố ý, có comment giải thích) — không sai spec về tinh thần |
| **Worker B6 - label voting** | OK | Vote theo tracklet, hysteresis 2-consecutive flip |

## Recommended approach

Chia làm **2 phase độc lập**, làm phase A trước (gain cao, blast radius nhỏ), phase B sau khi có log audit.

---

### Phase A — Bật LMBN backbone + fix L2-normalize

**A1. Copy weights và update env**
- File `lmbn_n_cuhk03_d.pth` ở project root đã có. Verify nó cũng nằm tại `src/inference_engine/src/assets/models/lmbn/lmbn_n_cuhk03_d.pth` (Plan agent đã thấy file, nhưng còn `lmbn_n_finetuned.pth` cùng folder — không dùng theo lựa chọn user).
- `src/inference_engine/.env`: set `INFERENCE_LMBN_WEIGHTS=/app/src/assets/models/lmbn/lmbn_n_cuhk03_d.pth`.
- `src/reid_worker/.env` (hoặc compose env): set `EMBEDDING_MODEL=lmbn` (kiểm tra prefix `REID_WORKER_` trong `src/reid_worker/src/core/config.py`).

**A2. Fix L2-normalize cho LMBN trong `model_registry.py`**
- 3 nhánh cần normalize sau forward, mirror OSNet:
  - `extract_embedding` (LMBN branch)
  - `extract_embedding_batch` (LMBN branch)
  - `extract_embedding_from_tensors` (LMBN branch)
- Pattern: `features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)`
- LMBN eval-mode output là `[B, 512, 7]` → `mean(dim=2)` ra `[B, 512]` (cùng dim OSNet → Qdrant không cần đổi schema).

**A3. Drop & recreate Qdrant collection `persons`**
- Vì collection cũ chứa OSNet vectors, trộn với LMBN vectors sẽ làm scoring loạn.
- Cách làm: 1-shot script gọi `qdrant_client.delete_collection("persons")` rồi để `_ensure_collection` tự tạo lại lúc worker khởi động. Có thể đặt thành Bash one-liner hoặc helper script.
- MongoDB persons history thì xoá tuỳ chọn (không bắt buộc, vì worker dùng Qdrant là source of truth cho matching; nhưng nếu giữ MongoDB persons cũ mà Qdrant rỗng → next run sẽ allocate id collision khi `id_allocator()` chạy. **Cần verify `id_allocator` đọc max(person_id) từ đâu**; nếu từ MongoDB thì OK, nếu từ Qdrant count thì cần đồng bộ wipe.).

**A4. KHÔNG tune `similarity_threshold` ngay**
- Giữ nguyên 0.73 ở lần chạy đầu. Ghi log score distribution rồi mới quyết định.
- Lý do: chưa biết score distribution của LMBN-CUHK03 trên domain của user — tune ngay là overfit.

---

### Phase B — Audit logic ReIDMatcher (làm sau khi Phase A đã chạy ≥2 video)

**B1. Wire `_record_decision` ra MongoDB (zero behavior change)**
- File: `src/reid_worker/src/matching/reid_matcher.py` constructor + `_record_decision` (line 285-286).
- Inject `decision_sink: Callable[[dict], None] | None` từ `workers/main.py`.
- Sink ghi 1 doc/decision vào collection mới `matcher_decisions` với schema: `{track_id, video_id, ts, method, source, similarity_score, runner_up_score, margin_to_runner_up, reuse_person_id, tentative_attempts, canonical_update_applied}`.
- Không thay đổi default value — chỉ thêm observability.

**B2. Audit script trên ≥3 video** (`scripts/audit_matcher_decisions.py`)
- Aggregate qua video:
  - Tỉ lệ `method`: `new_identity` / `gallery_match` / `tentative_*` / `soft_match_*` / `consistency_rejected` / `current_identity_maintained`.
  - Histogram `similarity_score` cho `gallery_match` vs `soft_match_at_promotion` vs `new_identity_blocked::gate_*`.
  - Diagnose ID-splitting: `new_identity` ratio cao + `near_gallery_deferred` cao + `gallery_match_below_soft_threshold` cao ⇒ threshold quá tight HOẶC embedding model quá yếu (so với Phase A baseline).
- **Output là input để quyết định bước B3-B6**. Không sửa threshold trước khi có data.

**B3. Xoá dead code (zero risk)**
- Remove tham số `good_streak_min_consecutive`, `good_streak_promotion_enabled` (không reference trong `match_tracklet`).
- Remove `update_sim_threshold` (comment `qdrant_store.py:171` xác nhận unused).
- Cleanup ở `reid_matcher.py`, `config.py`, `workers/main.py` wiring.

**B4. Đồng nhất defaults matcher ↔ config**
- 9 tham số bị lệch (xem bảng audit trong Plan agent output). Pick 1 nguồn truth (đề xuất: Settings là truth, matcher defaults align về Settings).
- Đặc biệt: `min_high_quality_frames` default 0 trong matcher (legacy) → đổi về 3, update test fixtures để pass explicit 0 khi cần.

**B5. Disable soft-match cascade theo feature flag (config-level rollback)**
- Thêm `enable_soft_match_paths: bool = False` trong `Settings`. Khi False: set `eager_soft_match_threshold = soft_match_threshold = 1.0` ở wiring.
- Giữ `tentative_fallback_enabled = True` (path này cần để xử lý borderline reentry).
- Rollback: env `REID_WORKER_ENABLE_SOFT_MATCH_PATHS=true` đưa về behavior cũ.
- Update test setup tương ứng — bật flag explicit ở các test cover soft path.

**B6. Đơn giản hoá `current_identity_switch_*` (3 magic → derive từ tham số khác)**
- Tại `reid_matcher.py:573-591`, thay 3 constant magic bằng:
  - `allow_gallery_switch = best_other.score ≥ similarity_threshold AND (best_other.score − current_score) ≥ match_margin AND current_score < current_identity_min_score`.
- Remove `current_identity_switch_min_score`, `current_identity_switch_min_margin`, `current_identity_switch_max_current_score`.
- Feature flag `legacy_current_identity_switch: bool = False` để rollback.

**B7. (deferred — chỉ làm nếu audit cho thấy cần) — review `near_gallery_defer_threshold`**
- Nếu `near_gallery_deferred` >15% decisions trong ≥1 video → defer band quá rộng, bias toward existing IDs.
- Action: chỉ áp dụng cho `track_id < 0` (untracked clusters) thay vì mọi tracklet.

---

## Critical files

- `src/inference_engine/src/core/config.py` (Settings — set `lmbn_weights`)
- `src/inference_engine/src/models/model_registry.py` (3 nhánh LMBN cần L2-norm, line 285-306 warmup; line 349-371, 390-391)
- `src/inference_engine/src/models/lightmbn_n.py` (verify forward output shape [B,512,7])
- `src/inference_engine/.env` + `src/reid_worker/.env` (env switches)
- `src/reid_worker/src/matching/reid_matcher.py` (28 params, `match_tracklet`, `_record_decision`)
- `src/reid_worker/src/matching/qdrant_store.py` (verify `update_sim_threshold` unused, drop collection trigger)
- `src/reid_worker/src/core/config.py` (Settings defaults, thêm `enable_soft_match_paths`, decision_sink)
- `src/reid_worker/src/workers/main.py` (wiring matcher, line 317-347; persistence callback line 5562-5643)
- `src/reid_worker/tests/test_qdrant_matcher.py` (sửa khi đổi default + feature flag)
- `src/deploy/docker-compose.yml` (verify bind mount `/app/src/assets`)
- `scripts/audit_matcher_decisions.py` (mới)

## Verification

1. **Inference engine standalone** (Phase A):
   ```
   cd src/inference_engine && uv run python -m src
   # Đợi log "model_registry.lmbn_loaded" + GET /readyz
   curl -F image1=@A1.jpg -F image2=@A2.jpg -F model=lmbn http://localhost:8000/similarity
   curl -F image1=@A1.jpg -F image2=@B1.jpg -F model=lmbn http://localhost:8000/similarity
   ```
   Kỳ vọng: same-person LMBN ≥ OSNet baseline; diff-person LMBN < same-person với margin rõ.

2. **E2E trên 2-3 video độc lập** (Phase A):
   - Drop Qdrant collection `persons`.
   - Chạy worker với `EMBEDDING_MODEL=lmbn` trên 2-3 video chưa từng dùng để tune.
   - Đếm số `person_id` mới so với ground-truth ID (đếm tay trên UI nếu chưa có ground-truth). **Không cherry-pick video**.

3. **Test suite**:
   ```
   uv run pytest src/reid_worker/tests/ -x
   uv run pytest src/inference_engine/tests/ -x
   ```
   Phase A nên zero break (dim 512 không đổi). Phase B sẽ cần update test theo từng bước B3-B6.

4. **Audit script** (Phase B):
   ```
   uv run python scripts/audit_matcher_decisions.py --videos video1,video2,video3
   ```
   Output bảng method distribution + similarity histogram qua video.

## Risks & rollback

- **Phase A**:
  - **Risk**: LMBN-CUHK03 chưa fine-tune trên domain user → score distribution có thể thấp hơn OSNet, threshold 0.73 cứng nhắc có thể gây tăng `new_identity`. **Mitigation**: nếu audit cho thấy phần lớn `gallery_match` rơi vào 0.65-0.72 → consider giảm threshold sau khi đo, KHÔNG đoán trước.
  - **Risk**: nếu thiếu A2 (L2-norm fix), Qdrant scoring vẫn đúng (COSINE) nhưng EMA + heuristics dùng vector raw → drift. **Mitigation**: A2 bắt buộc, không skip.
  - **Rollback**: set `EMBEDDING_MODEL=osnet` về OSNet; drop Qdrant collection lại nếu cần.
- **Phase B**:
  - **Risk**: simplify quá sớm khi chưa có audit data → có thể đẩy ID-splitting sang chiều ngược (under-creation). **Mitigation**: tuân thứ tự B1 → B2 (audit) → B3 (dead code, an toàn) → B4-B6 (config-flagged).
  - **Rollback**: mọi behavior change ở B5-B6 đều đi qua feature flag config, có thể tắt bằng env var không cần redeploy.

## Open questions (cần xác nhận khi bắt đầu execute)

1. `id_allocator()` đọc max(person_id) từ MongoDB hay Qdrant? Quyết định có cần wipe MongoDB `persons` khi drop Qdrant collection.
2. Có ground-truth IDs từ video test cũ để đo objective trước/sau, hay sẽ đánh giá bằng mắt trên UI?
3. Phase B có thể chạy song song với production data collection không (cần production ổn để có log đủ aggregate)?
