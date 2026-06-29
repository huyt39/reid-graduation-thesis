from __future__ import annotations

import threading
import time

import cv2
import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.config import settings

cv2.setNumThreads(1)

log = structlog.get_logger()

_sources = settings.source_map()
_latest: dict[str, bytes] = {}
_locks: dict[str, threading.Lock] = {d: threading.Lock() for d in _sources}


_state_lock = threading.Lock()
_refcount: dict[str, int] = {d: 0 for d in _sources}
_stop: dict[str, threading.Event] = {}
_threads: dict[str, threading.Thread] = {}

_ended: dict[str, bool] = {d: False for d in _sources}
_replay: dict[str, threading.Event] = {}


def _downscale(frame):
    max_dim = int(settings.max_dim)
    if max_dim <= 0:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame
    s = max_dim / float(longest)
    return cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def _set_ended(device_id: str, value: bool) -> None:
    with _state_lock:
        _ended[device_id] = value


def _reader(device_id: str, source: str, stop_event: threading.Event,
            replay_event: threading.Event) -> None:

    quality = int(settings.jpeg_quality)
    src = int(source) if source.isdigit() else source
    log.info("raw_stream.reader_started", device_id=device_id)
    while not stop_event.is_set():
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.warning("raw_stream.open_failed", device_id=device_id, source=source)
            time.sleep(0.5)
            continue
        _set_ended(device_id, False)
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        send_fps = min(float(settings.fps), src_fps)
        send_interval = 1.0 / send_fps if send_fps > 0 else 1.0 / 10.0
        start = time.perf_counter()
        fidx = 0
        last_send_vt = -1.0
        while not stop_event.is_set():
            if not cap.grab():
                break
            fidx += 1
            video_t = fidx / src_fps
            real_t = time.perf_counter() - start
            if video_t > real_t:
                time.sleep(min(video_t - real_t, 0.1))
            if video_t - last_send_vt < send_interval:
                continue
            last_send_vt = video_t
            ok, frame = cap.retrieve()
            if not ok:
                continue
            ok, buf = cv2.imencode(".jpg", _downscale(frame), [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                with _locks[device_id]:
                    _latest[device_id] = buf.tobytes()
        cap.release()
        if stop_event.is_set():
            break

        _set_ended(device_id, True)
        log.info("raw_stream.reader_eof_hold", device_id=device_id)
        while not stop_event.is_set():
            if replay_event.is_set():
                replay_event.clear()
                _set_ended(device_id, False)
                log.info("raw_stream.reader_replay", device_id=device_id)
                break  # re-open the capture from frame 0 on the next outer loop
            time.sleep(0.1)
    log.info("raw_stream.reader_stopped", device_id=device_id)


def _acquire(device_id: str) -> None:
    with _state_lock:
        _refcount[device_id] += 1
        t = _threads.get(device_id)
        if t is None or not t.is_alive():
            ev = threading.Event()
            ev_replay = threading.Event()
            _stop[device_id] = ev
            _replay[device_id] = ev_replay
            _ended[device_id] = False
            t = threading.Thread(target=_reader,
                                 args=(device_id, _sources[device_id], ev, ev_replay),
                                 name=f"raw-{device_id}", daemon=True)
            _threads[device_id] = t
            t.start()


def _release(device_id: str) -> None:
    with _state_lock:
        _refcount[device_id] = max(0, _refcount[device_id] - 1)
        if _refcount[device_id] == 0:
            ev = _stop.get(device_id)
            if ev is not None:
                ev.set()
            _threads[device_id] = None
            _replay.pop(device_id, None)
            _ended[device_id] = False
            with _locks[device_id]:
                _latest.pop(device_id, None)


def _mjpeg_generator(device_id: str):
    interval = 1.0 / float(settings.fps) if settings.fps > 0 else 1.0 / 10.0
    boundary = b"--frame\r\n"
    _acquire(device_id)
    try:
        while True:
            with _locks[device_id]:
                data = _latest.get(device_id)
            if data is not None:
                yield boundary + b"Content-Type: image/jpeg\r\nContent-Length: " \
                    + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
            time.sleep(interval)
    finally:
        _release(device_id)


app = FastAPI(title="raw_stream")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    log.info("raw_stream.started", devices=list(_sources.keys()), port=settings.port,
             fps=settings.fps, max_dim=settings.max_dim, lazy=True)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "devices": list(_sources.keys())}


@app.get("/readyz")
def readyz() -> dict:
    return {"status": "ok"}


@app.get("/devices")
def devices() -> dict:
    return {"devices": list(_sources.keys())}


@app.get("/status")
def status(device_id: str = Query(...)) -> dict:
    if device_id not in _sources:
        raise HTTPException(status_code=404, detail=f"unknown device_id {device_id}")
    with _state_lock:
        ended = _ended.get(device_id, False)
    return {"device_id": device_id, "ended": ended}


@app.post("/replay")
def replay(device_id: str = Query(...)) -> dict:
    if device_id not in _sources:
        raise HTTPException(status_code=404, detail=f"unknown device_id {device_id}")

    with _state_lock:
        ev = _replay.get(device_id)
        if ev is not None:
            ev.set()
            _ended[device_id] = False
    return {"ok": ev is not None}


@app.get("/mjpeg")
def mjpeg(device_id: str = Query(...)) -> StreamingResponse:
    if device_id not in _sources:
        raise HTTPException(status_code=404, detail=f"unknown device_id {device_id}")
    return StreamingResponse(
        _mjpeg_generator(device_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
