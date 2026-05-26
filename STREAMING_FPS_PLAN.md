# Plan v6: Fix letterbox/aspect flicker — LiveFeed Card stretched by PersonsPanel

## Context (v6 — current)

User báo: "ở một số frame xuất hiện viền đen ở trên hoặc dưới, khiến kích thước frame thu nhỏ lại hoặc to ra". Đây **không phải FPS giật** (FPS đã ok ở v5a), mà là **aspect ratio của khung hiển thị thay đổi** giữa các frame ⇒ `object-contain` của `<img>` tạo letterbox khác nhau.

### Root cause (đã verify đọc 2 file)

**`src/ui/app/live/_components/live-view.tsx:99`** — layout cha:
```tsx
<div className="flex flex-col lg:flex-row gap-4">
  <PersonsPanel ... />
  <LiveFeed frame={...} ... />
</div>
```
Không có `items-start` ⇒ default `align-items: stretch` ⇒ LiveFeed Card bị **stretch height** theo sibling cao nhất.

**`src/ui/app/live/_components/persons-panel.tsx:72`** — sibling PersonsPanel:
```tsx
<aside className="w-full lg:w-72 shrink-0 flex flex-col gap-3">
```
Width cố định `w-72` (288px) ⇒ LiveFeed width KHÔNG đổi. Nhưng aside KHÔNG có `max-h` ⇒ height tăng theo số person cards (inner `overflow-y-auto` ở line 78 không hoạt động vì outer không có constraint chiều cao).

**`src/ui/app/live/_components/live-feed.tsx:135`** — LiveFeed Card:
```tsx
<Card className="flex-1 relative overflow-hidden bg-black p-0 min-h-[400px]">
  <img className="w-full h-full object-contain" ... />
```
`flex-1 min-h-[400px]` không cố định aspect. Height = `max(400px, stretched-by-sibling)`.

**Chuỗi nhân quả**: nhiều người vào cảnh ⇒ PersonsPanel cao hơn ⇒ stretch LiveFeed Card cao hơn ⇒ Card aspect ratio (W/H) giảm ⇒ `<img>` (video 16:9) `object-contain` ⇒ width bị giới hạn bởi height, image scale down ⇒ **viền đen trên/dưới to ra, frame thu nhỏ**. Khi người rời cảnh ⇒ ngược lại.

Secondary cause (chưa chắc có ảnh hưởng nhưng đáng note): raw stream (size 720, encode_preview_max=720) và processed stream (native size, worker không resize) có dimensions khác nhau. UI ưu tiên rawFrame nên khi cả 2 có frame, baseFrame luôn = raw ⇒ KHÔNG switch ⇒ secondary cause không trigger trong flow bình thường. Bỏ qua pha này.

### Fix

File: `src/ui/app/live/_components/live-feed.tsx`

Đổi:
```tsx
<Card className="flex-1 relative overflow-hidden bg-black p-0 min-h-[400px]">
```
thành:
```tsx
<Card className="flex-1 relative overflow-hidden bg-black p-0 aspect-video">
```

`aspect-video` (= 16/9) ⇒ Card height = width × 9/16, **không còn phụ thuộc PersonsPanel**. Image `object-contain` 16:9 video sẽ fit perfectly không letterbox.

Bonus (cũng nhỏ, gộp luôn): thêm `items-start` vào parent flex ở `live-view.tsx:99` để defensive — phòng trường hợp aspect-video không đủ (vd dùng grid layout sau).

File: `src/ui/app/live/_components/live-view.tsx`

```diff
- <div className="flex flex-col lg:flex-row gap-4">
+ <div className="flex flex-col lg:flex-row gap-4 lg:items-start">
```

### Trade-off + note

- `aspect-video` giả định video 16:9. Nếu user dùng video 4:3 hoặc dọc thì sẽ có letterbox **consistent** (không nhấp nháy). Không phải xấu — vẫn ổn định.
- Nếu user muốn aspect động theo video, có thể đo `img.naturalWidth/naturalHeight` ở first load và set inline style `aspectRatio`. Phức tạp hơn, làm sau nếu cần.
- PersonsPanel height vẫn growing — sẽ bị overflow vertically dưới Card. Cosmetic issue khác, có thể fix riêng bằng `max-h-[80vh] overflow-y-auto` trên aside. Note nhưng không bắt buộc pha này.

## Critical files

- `src/ui/app/live/_components/live-feed.tsx` — thay `min-h-[400px]` bằng `aspect-video`.
- `src/ui/app/live/_components/live-view.tsx` — thêm `lg:items-start`.

## Verification

1. Rebuild UI: `docker compose -f src/deploy/docker-compose.yml up -d --build ui` (UI rebuild ~30s).
2. Mở `http://localhost:3000/live`. Quan sát:
   - LiveFeed Card luôn 16:9 aspect.
   - Khi người vào/ra cảnh, **kích thước frame không đổi**, viền đen (nếu có) constant.
   - UI badge FPS vẫn hiển thị bình thường.
3. Test ≥2 video khác nhau (vắng + đông người) để verify stability.

## Out of scope

- Resize processed stream về 720 ở worker (giải quyết cause secondary).
- PersonsPanel max-h + scroll (cosmetic riêng).
- Async capture loop pha 5b (đã out of scope từ v5 vì user đã đạt mục tiêu FPS).

---

# Plan v5: Tune YOLO/encode env (pha 5a) — refactor async capture conditional (pha 5b) (đã apply)

## Context (v5 — current)

Sau v4, telemetry xác nhận:
- Edge `processed_frames/frame_idx = 96.7%` ✓ heartbeat lọt qua đúng
- Edge `fps=6.43 ≈ source_fps=6.65` ✓ — fix throttle hoạt động
- UI `recv 5` khớp với edge fps. UI/streaming KHÔNG còn là bottleneck.

**Bottleneck duy nhất còn lại = edge source_fps thấp** (6.65 fps trên Mac CPU). Chi tiết:
- `avg_detect_ms=117ms` (YOLO mỗi 2 frame).
- Lý thuyết per-2-frame-cycle: YOLO 117 + 2× cv2.read + 2× encode 12 + 2× publish 4 ≈ 165ms ⇒ ~12 fps lý thuyết.
- Thực tế 6.65 fps ⇒ có overhead ~135ms/cycle unexplained (cv2.VideoCapture decode H264 trên frame to + Kafka producer batch + thread block).

User chọn "cả hai": tune env trước (pha 5a), refactor async capture sau nếu chưa đủ (pha 5b).

## Pha 5a: Tune env (không sửa code)

**Mục tiêu**: source_fps lên ~18-20 fps để UI smooth nhìn rõ rệt. Trade-off: YOLO nhỏ hơn có thể miss person nhỏ/xa (cần verify bằng visual + số tracklet).

Chỉnh `src/edge/.env`:

```diff
- EDGE_YOLO_IMGSZ=320
+ EDGE_YOLO_IMGSZ=192               # YOLO ~2.5x nhanh hơn

- EDGE_DETECT_EVERY_N_FRAMES=2
+ EDGE_DETECT_EVERY_N_FRAMES=3      # YOLO chạy 1/3 frame (~3-5 fps detection rate)

- EDGE_MAX_ENCODE_DIM=0
+ EDGE_MAX_ENCODE_DIM=720           # downscale frame trước encode (giảm encode time + bandwidth)
```

(`EDGE_MAX_ENCODE_DIM` đã được implement trong `_prepare_outbound_frame` ở `src/edge/src/workers/main.py:133-161`, scale theo longest dimension và scale bbox tương ứng.)

Cách restart **không cần build lại** (env_file load runtime):

```bash
docker compose -f src/deploy/docker-compose.yml up -d --force-recreate edge
```

### Kỳ vọng pha 5a

- Edge `source_fps ≈ 15-20`, `fps` cũng tương đương (>95% ratio).
- UI badge `recv ≈ 15-20`.
- `avg_detect_ms` giảm xuống ~50-70ms.
- `avg_encode_ms` giảm xuống ~5-7ms (frame đã downscale).
- Tradeoff: detection ở imgsz 192 có thể miss small persons; verify bằng số tracklet/person_id mới (so v3 baseline).

Sau khi đo, user paste:
- UI badge
- `edge_progress` log (kỳ vọng fps tăng đáng kể)
- `streaming.fps_summary` log

→ Nếu `recv ≥ 15` và visual smooth ⇒ DONE, không cần pha 5b.
→ Nếu `recv < 15` HOẶC accuracy giảm không chấp nhận được ⇒ pha 5b.

## Pha 5b (conditional): Async capture loop

**Khi nào cần**: nếu pha 5a chưa đủ smooth, hoặc user muốn giữ accuracy gốc (imgsz=320, N=2).

**Spec**:

Tách `EdgePipeline.run()` ở `src/edge/src/workers/main.py:163-373` thành 2 thread:
1. **Capture thread** (daemon): vòng lặp `cv2.read()` liên tục. Ghi `latest_frame, latest_frame_idx` vào shared state có `threading.Lock`. Drop old frame (chỉ giữ latest).
2. **Detection thread** (main, async loop): tick ở rate cố định (vd 30 fps). Mỗi tick:
   - Đọc `latest_frame` snapshot từ shared state.
   - Nếu `latest_frame_idx % N == 0` ⇒ chạy YOLO, build outbound_detections.
   - Encode + publish với `frame_number = latest_frame_idx` và detections (rỗng nếu skip YOLO).
   - Skip nếu `latest_frame_idx` chưa đổi từ lần publish trước (tránh dup).

Lợi ích: cv2.read không còn block detection; source_fps lên native video FPS (25-30). YOLO vẫn chạy ~5-8 fps cho frame có detection.

Concern + giải pháp:
- **Frame ordering**: capture thread tăng frame_idx monotonic ⇒ publish thứ tự được giữ.
- **Lost frames**: nếu capture > publish rate, một số frame bị drop (chỉ keep latest). OK cho preview, không OK cho ReID accuracy. Nhưng vì YOLO chỉ chạy mỗi N-frame, không cần mọi frame qua YOLO.
- **Thread safety**: Python GIL + lock cho 1 frame shared state ⇒ minimal.
- **Test**: cần verify edge tests vẫn pass (chủ yếu test logic scoring, không depend thread).

**Critical files (pha 5b)**:
- `src/edge/src/workers/main.py` — refactor `run()` thành 2 thread.

**Verification (pha 5b)**:
- Source_fps kỳ vọng = native video FPS (25-30).
- UI badge `recv` ≈ source_fps.
- Tracklet quality không bất thường vs baseline.
- `pytest src/edge/tests/` pass.

## Critical files (tổng)

- `src/edge/.env` — 3 env vars (pha 5a).
- (Conditional) `src/edge/src/workers/main.py` — async capture refactor (pha 5b).

## Verification flow

1. Pha 5a: edit `.env` + restart edge (1 command). Đo. Quyết tiếp.
2. Nếu cần pha 5b: refactor + rebuild edge. Đo lại.

## Out of scope

- Tách topic `edge_preview` riêng.
- Đổi YOLO model (yolo11n vs best_26).
- Triton GPU.
- WebRTC.

---

# Plan v4: Fix heartbeat throttle bug (real publish "đầu độc" heartbeat ngay sau) (đã apply)

## Context (v4 — current)

Telemetry v3 đã hoạt động. Số đo từ user:

```
UI badge: recv 4–5 • render 38–45 • age 316–443ms
Edge:     processed_frames=575, frame_idx=958 → 60% publish ratio
          fps=4.69, source_fps=7.81, avg_detect_ms=170
Streaming raw: consumed_fps=4.96 (≈ edge fps)
Streaming processed: consumed_fps=2.01
```

Edge v2 ĐÃ active (skip_detection flag working) nhưng tỷ lệ publish vẫn chỉ ~60% thay vì ~100%. **Đây là một bug nữa mà v2 đã bỏ sót**.

### Root cause (đã đọc `src/edge/src/workers/main.py:319-326`)

```python
is_heartbeat = not outbound_detections
if is_heartbeat and self._preview_min_interval_s > 0:
    now = time.perf_counter()
    if (now - self._last_preview_publish_at) < self._preview_min_interval_s:
        continue
    self._last_preview_publish_at = now
else:
    self._last_preview_publish_at = time.perf_counter()  # ← BUG
```

Logic intent: throttle heartbeat ở 30 fps (`_preview_min_interval_s = 1/30 ≈ 33ms`) để chống ngập Kafka khi source > 30 fps.

Vấn đề: nhánh `else` (real publish) cũng reset `_last_preview_publish_at = now`. Vì YOLO frame và heartbeat frame xen kẽ (N=2), trình tự thực tế:
1. YOLO frame: read (~5ms) + YOLO (~170ms) + publish + **reset `_last_preview = now`**.
2. Skip frame ngay sau: read (~5ms) + heartbeat check: `now - _last_preview ≈ 5–10ms < 33ms` ⇒ **SKIP**.
3. YOLO frame tiếp theo: publish + reset.
4. Skip frame: bị block lần nữa.

⇒ Mỗi YOLO frame "đầu độc" heartbeat ngay sau nó ⇒ chỉ YOLO frames publish ⇒ ratio = ~50%. Lý thuyết source_fps = 1 / 90ms = ~11 fps; thực tế 7.81 vì Kafka publish có overhead variable.

### Fix

Tách `_last_heartbeat_at` riêng. Real publish KHÔNG touch biến này. Throttle vẫn áp dụng đúng (mục đích: 2 heartbeat liên tiếp cách nhau ≥ 33ms), nhưng không bị real publish reset.

File: `src/edge/src/workers/main.py`

```python
# __init__:
self._last_heartbeat_at: float = 0.0  # rename + chỉ heartbeat update

# trong run():
is_heartbeat = not outbound_detections
if is_heartbeat and self._preview_min_interval_s > 0:
    now = time.perf_counter()
    if (now - self._last_heartbeat_at) < self._preview_min_interval_s:
        continue
    self._last_heartbeat_at = now
# bỏ else branch — real publish không liên quan tới heartbeat throttle
```

Bonus cosmetic (không bắt buộc, để pha sau nếu muốn): `live-view.tsx:41-45` tạo `{...baseFrame, tracked_persons: hybridTrackedPersons}` mỗi render ⇒ `frame` props identity change ⇒ `useEffect` ở `live-feed.tsx` fire dù `frame_number` không đổi ⇒ render counter inflated (giải thích `render 38-45` mà `recv 4-5`). Fix bằng `useMemo` dep theo `frame_number`. Không ảnh hưởng performance thực, chỉ làm telemetry chính xác hơn.

## Critical files

- `src/edge/src/workers/main.py` — rename + tách throttle state.

## Verification

1. `docker compose -f src/deploy/docker-compose.yml up -d --build --force-recreate edge` (chỉ build edge, ~1 phút).
2. Đợi ~30s edge replay video.
3. Đọc lại 3 số:
   - UI badge: kỳ vọng `recv ≈ 7-8` (≈ source_fps).
   - Edge log: `processed_frames / frame_idx` kỳ vọng ≈ 95-100%; `fps ≈ source_fps`.
   - Streaming raw: `consumed_fps ≈ 7-8`.
4. Quality regression: số tracklet/person_id mới không bất thường (worker vẫn early-return cho heartbeat).

## Sau v4 — nếu vẫn muốn smooth hơn

User hiện chọn "chỉ fix throttle, tune sau". Nếu recv vẫn chưa đủ smooth (8 fps < 15-30 fps mong muốn), lựa chọn pha sau (chưa làm):
- Giảm `EDGE_YOLO_IMGSZ` 320 → 192/224 (YOLO ~2.5x faster, mất precision nhỏ).
- Async capture loop (decouple cv2.read khỏi YOLO; source_fps lên native ~25-30 fps).
- Đổi sang YOLO model nhẹ hơn (yolo11n.pt vs best_26.pt).

## Out of scope (lần này)

- Tách topic `edge_preview` riêng.
- Optimize YOLO inference time.
- Fix render counter inflated ở UI (cosmetic).

---

# Plan v3: Fix telemetry (badge=0) + hướng dẫn diagnose tiếp

## Context (Plan v3 — current)

Sau khi apply v2, user chạy lại và báo:
1. Video **vẫn giật và lag**.
2. Badge FPS góc trái dưới **chỉ hiển thị `recv 0 • render 0 • age 0ms`**.

**Root cause của badge=0** (đã verify, đọc `src/ui/app/live/_components/live-feed.tsx:131-143`):

```tsx
useEffect(() => {
  const timer = setInterval(() => { ... }, 1000);
  return () => clearInterval(timer);
}, [frame]);  // ← BUG: dep [frame] ⇒ interval bị clear+recreate mỗi khi frame thay đổi
```

Vì WebSocket đẩy frame mới mỗi ~100ms, `useEffect` cleanup chạy và clear timer **trước khi 1 giây trôi qua**. Callback `setInterval` không bao giờ fire ⇒ counters `recvCountRef`/`renderCountRef` không bao giờ được đọc + reset ⇒ state luôn = `{recv:0, render:0, ageMs:0}`. `ageMs=0` cũng vì state `fpsStats` không bao giờ được update.

→ **Telemetry không nói lên gì về performance thực**. Phải fix telemetry trước, rồi mới có cơ sở quyết định bottleneck tiếp theo.

Bonus: `frame.created_at` từ Avro là `time.time_ns()` (nano), code đã chia `/1e6` đúng ⇒ logic age đúng, chỉ là không chạy.

## Approach v3 (1 fix duy nhất + hướng dẫn diagnose)

### 1. Refactor setInterval cho stable timer

File: `src/ui/app/live/_components/live-feed.tsx:98-143`

Pattern chuẩn: dùng `frameRef` mutable ref để setInterval đọc latest frame mà không cần dep `[frame]`.

```tsx
const frameRef = useRef(frame);
useEffect(() => { frameRef.current = frame; }, [frame]);

useEffect(() => {
  const timer = setInterval(() => {
    const f = frameRef.current;
    const recv = recvCountRef.current;
    const render = renderCountRef.current;
    recvCountRef.current = 0;
    renderCountRef.current = 0;
    const ageMs = f
      ? Math.max(0, Math.round(Date.now() - f.created_at / 1e6))
      : 0;
    setFpsStats({ recv, render, ageMs });
  }, 1000);
  return () => clearInterval(timer);
}, []);  // ← mount once, never recreated
```

Không thay đổi 2 useEffect khác (effect tăng recvCount và effect drawOverlay) — chúng dùng dep `[frame]` đúng mục đích (fire mỗi frame mới).

### 2. Hướng dẫn diagnose sau khi telemetry hoạt động

Sau rebuild, user mở UI, đọc badge. Ba kịch bản sẽ giúp xác định bottleneck:

| Triệu chứng badge | Bottleneck | Action |
|---|---|---|
| `recv < 10`, `age < 300ms` | Edge publish chậm (YOLO bottleneck vẫn còn) | Check log `edge_progress.fps`; nếu `fps ≈ source_fps` rồi mà vẫn thấp ⇒ source_fps thấp do cv2 đọc chậm; nếu `fps << source_fps` ⇒ fix v2 chưa active (container chưa rebuild). |
| `recv ≥ 12`, `render << recv` | UI rendering bottleneck (canvas/decode) | Kiểm tra frame resolution; có thể downscale ở edge. |
| `recv ≥ 12`, `render ≈ recv`, `age > 1000ms` | Backlog Kafka/streaming | Check log `streaming.fps_summary`; nếu `consumed_fps >> broadcast_fps` ⇒ broadcast_max_fps thấp; nếu `consumed_fps` thấp ⇒ consumer poll chậm. |
| Cả `recv`, `render` cao + `age` thấp + vẫn cảm giác lag | Có thể là hybrid raw+processed IoU matching trong `live-view.tsx` | Tạm thời tắt 1 trong 2 stream để confirm. |

User cần paste lại số liệu badge + `edge_progress` log + `streaming.fps_summary` log để mình tune tiếp.

## Critical files

- `src/ui/app/live/_components/live-feed.tsx` — refactor 1 useEffect (lines 131-143) thành 2 useEffect với frameRef pattern.

Không động vào edge/streaming/worker.

## Verification

1. `./scripts/demo.sh up --build` (chỉ build UI, edge/streaming/worker không thay đổi lần này).
2. Mở `http://localhost:3000/live`, chờ vài giây.
3. Badge FPS phải hiển thị số khác 0 (vd `recv 8 • render 8 • age 250ms`).
4. Paste số liệu + log để mình quyết định tune tiếp.

## Out of scope (lần này)

- Tune edge YOLO / preview throttle / frame skip — sẽ làm sau khi có số liệu thật.
- Tách topic `edge_preview` riêng (pha 3 nếu cần).

---

# Plan v2 (đã apply, giữ lại để tham khảo)

## Context

Sau khi đã apply plan v1 (bỏ skip frame rỗng, bỏ double-encoding, nâng broadcast/UI FPS cap), user chạy lại stack thấy video **vẫn giật và lag**. Đọc log thực tế từ `edge_progress`:

```
fps: 7.22–7.87           ← edge publish tới Kafka
source_fps: 14.44–15.74  ← cv2.VideoCapture đọc từ video
avg_detect_ms: 93–103    ← YOLO inference (BOTTLENECK)
avg_encode_ms: 5.7
avg_publish_ms: 0.4
avg_outbound_detections: 0.45–0.69
```

Config trong `src/edge/.env`: `EDGE_DETECT_EVERY_N_FRAMES=2`, `EDGE_YOLO_IMGSZ=320`, model `best_26.pt`. Tức là YOLO đã skip 1/2 frames và imgsz nhỏ, **nhưng vẫn 100ms/inference** trên Mac CPU.

**Root cause kỹ thuật** (mới phát hiện): `src/edge/src/workers/main.py:213-218` — khi `detect_every_n_frames > 1` và `frame_idx % N != 0`, code `continue` ngay ⇒ **frame skip YOLO bị bỏ hoàn toàn, không encode/publish**. Tức là pipeline đang publish ở tốc độ YOLO (~7 fps) thay vì tốc độ capture (~15 fps).

Có config dead trong `.env` (`EDGE_PREVIEW_ENABLED=true`, `EDGE_PREVIEW_TOPIC=edge_preview`, `EDGE_PREVIEW_FPS=12`) và topic `edge_preview` trong `scripts/demo.sh` — nhưng **chưa có code implement**. Đây là tàn dư từ planning trước đó. Pha này không động vào.

Mục tiêu pha 2: stream lên ngang `source_fps` (~15 fps trên video hiện tại) bằng cách tách publish-rate khỏi YOLO-rate. Bonus: thêm telemetry để biết chính xác FPS ở mỗi tầng cho lần tune tiếp theo.

## Approach

### 1. Edge: frame skip YOLO vẫn publish heartbeat

File: `src/edge/src/workers/main.py:213-225`

Hiện tại 2 chỗ `continue`:
```python
# skip-by-N
if detect_every_n_frames > 1 and frame_idx % detect_every_n_frames != 0:
    continue
# pre-skipper similarity check
if detect_every_n_frames <= 1 and not pre_skipper.should_process(frame):
    continue
```

Đổi thành: thay vì `continue`, đặt cờ `skip_detection = True`, **bỏ qua YOLO** nhưng vẫn rơi xuống nhánh encode+publish với `detections = []`. Reuse logic heartbeat đã có ở plan v1.

Pseudo:
```python
skip_detection = (
    (detect_every_n_frames > 1 and frame_idx % detect_every_n_frames != 0)
    or (detect_every_n_frames <= 1 and not pre_skipper.should_process(frame))
)
if skip_detection:
    detections = []
else:
    detections = self.detector.infer(frame)
    ... pre_skipper.update_after_detection(detections) ...

# tiếp tục như cũ: compute scores nếu có detections, build outbound_detections,
# rồi heartbeat throttle check + encode + publish (logic đã có sẵn từ plan v1).
```

Kết quả: edge publish ~`source_fps` (~15 fps). YOLO vẫn chạy ~7 fps (đủ cho ReID). Worker đã có `if not detections: return` (main.py:1738) — heartbeat tự skip.

Lưu ý quan trọng: `preview_throttle_fps` (default 30) đã có sẵn ở plan v1, sẽ chặn khi heartbeat vượt 30 fps. Giữ nguyên.

### 2. Streaming: periodic broadcast counter log

File: `src/streaming/src/workers/kafka_loop.py`

Thêm vào `run_kafka_loop`:
- Counter `messages_consumed`, `broadcasts_sent` per device
- Mỗi 5s log `streaming.fps_summary` với rate/device
- Tăng counter chỗ poll messages và chỗ schedule broadcast

Không thay đổi behavior; chỉ là quan sát.

### 3. UI: FPS overlay trên live-feed

File: `src/ui/app/live/_components/live-feed.tsx`

Thêm 1 badge nhỏ ở góc trái dưới hiển thị:
- `recv` FPS (số frame nhận từ WS trong 1s gần nhất)
- `render` FPS (số lần `drawOverlay` chạy)
- `age` (ms) = now - `frame.created_at` ⇒ trễ end-to-end

Implement bằng ref counter + setInterval 1s update state. Luôn-on (lightweight) để dễ debug; có thể wrap behind `process.env.NEXT_PUBLIC_DEBUG_FPS` nếu prefer ẩn.

### 4. Verify + tune

Sau khi rebuild + restart:
1. Quan sát badge FPS trên UI → expect `recv ~12-15 fps` (gần `source_fps` edge).
2. Quan sát log streaming `streaming.fps_summary` → expect tương đương.
3. Quan sát log edge `edge_progress` → `fps` vẫn ~7-8 (YOLO rate, không thay đổi), nhưng cần thêm metric "published_total / runtime" nếu chưa có — fps hiện tại đang đo published_messages, sẽ tăng vì giờ count cả heartbeat. Đây là điều tốt, nhưng cần biết để không bị nhầm.
4. Nếu UI vẫn giật ⇒ check:
   - `age` lớn (> 500ms): network/queue lag → tune Kafka consumer
   - `recv` thấp (< source_fps): edge publish chưa đủ → check log edge
   - `render` thấp hơn `recv` nhiều: UI bottleneck → check decode JPEG hoặc canvas redraw

## Critical files

- `src/edge/src/workers/main.py` — refactor `continue` skip logic thành `skip_detection` flag.
- `src/streaming/src/workers/kafka_loop.py` — counter + periodic log.
- `src/ui/app/live/_components/live-feed.tsx` — FPS overlay badge.

Không tạo file mới. Không đổi schema/config defaults.

## Verification

1. `./scripts/demo.sh up --build` (không cần `--reset`, giữ data ReID baseline để so).
2. Mở `http://localhost:3000/live`, chọn device.
3. Đọc badge FPS overlay; expect `recv ≥ 12 fps`, `age < 500 ms`.
4. Đọc `docker compose -f src/deploy/docker-compose.yml logs streaming --tail 50 | grep fps_summary`; expect tương đương `recv`.
5. Test ≥2 video khác nhau (vắng + đông người) — kết luận chỉ rút sau khi có số liệu của cả 2.
6. Chạy `pytest` ở `src/edge`, `src/streaming` — fix test nào tham chiếu `continue` cũ hoặc signature `kafka_loop`.
7. Quality regression: số tracklet/person_id mới so với baseline trước fix ⇒ không bất thường (heartbeat không nên ảnh hưởng vì worker early-return).

## Out of scope

- Implement topic `edge_preview` riêng (cleaner, nhưng cần schema mới + producer thứ 2 ở edge + consumer mới ở streaming). Ghi nhận làm pha 3 nếu user muốn.
- Tối ưu YOLO inference (giảm imgsz xuống 192, dùng yolo11n.pt thay best_26.pt, hoặc đẩy lên Triton GPU). Mất accuracy detection, cần benchmark riêng.
- WebRTC theo diagram, binary WebSocket, downscale frame ở edge.
