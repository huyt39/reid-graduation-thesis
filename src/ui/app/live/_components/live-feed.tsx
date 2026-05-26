"use client";

import { useEffect, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import { getLiveStatusLabel } from "@/lib/reid-evidence";
import type { FrameUpdate, TrackedPerson } from "@/hooks/use-websocket";

function visColor(score: number): string {
  if (score >= 0.7) return "#22c55e";
  if (score >= 0.4) return "#eab308";
  return "#ef4444";
}

// Golden-ratio hue spacing — gives visually distinct, stable colors per person_id.
// Returns hex (not hsl()) so the `color + "18"` / `color + "cc"` alpha-hex
// concatenation downstream produces a valid CSS color.
function personColor(personId: number): string {
  const hue = (personId * 137.508) % 360;
  const s = 0.7;
  const l = 0.55;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if (hue < 60) [r, g, b] = [c, x, 0];
  else if (hue < 120) [r, g, b] = [x, c, 0];
  else if (hue < 180) [r, g, b] = [0, c, x];
  else if (hue < 240) [r, g, b] = [0, x, c];
  else if (hue < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  const toHex = (v: number) =>
    Math.round((v + m) * 255).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function drawOverlay(canvas: HTMLCanvasElement, img: HTMLImageElement, persons: TrackedPerson[]) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const W = canvas.width;
  const H = canvas.height;

  for (const p of persons) {
    const [x1, y1, x2, y2] = p.bbox;
    const isNorm = x2 <= 1.0 && y2 <= 1.0;
    const rx1 = isNorm ? x1 * W : x1;
    const ry1 = isNorm ? y1 * H : y1;
    const rx2 = isNorm ? x2 * W : x2;
    const ry2 = isNorm ? y2 * H : y2;
    const bw = rx2 - rx1;
    const bh = ry2 - ry1;

    const isTentative = p.tracklet_state === "tentative";
    const status = p.status ?? (isTentative ? "tentative" : "confirmed");
    const color =
      p.person_id != null
        ? personColor(p.person_id)
        : status === "recovering"
          ? "#f59e0b"
          : isTentative
            ? "#94a3b8"
            : visColor(p.live_visibility_score);

    ctx.strokeStyle = color;
    ctx.lineWidth = isTentative ? 1.5 : 2;
    ctx.setLineDash(isTentative ? [6, 3] : []);
    ctx.strokeRect(rx1, ry1, bw, bh);
    ctx.setLineDash([]);

    ctx.fillStyle = color + "18";
    ctx.fillRect(rx1, ry1, bw, bh);

    const label = isTentative
      ? "?"
      : p.person_id === null
        ? `raw ${Math.max(p.confidence, p.live_visibility_score).toFixed(2)}`
        : `#${p.person_id} ${getLiveStatusLabel(status)} ${p.live_visibility_score.toFixed(2)}`;
    ctx.font = "600 12px sans-serif";
    const tw = ctx.measureText(label).width;
    const labelY = ry1 > 18 ? ry1 - 4 : ry1 + 16;
    ctx.fillStyle = color + "cc";
    ctx.fillRect(rx1, labelY - 13, tw + 8, 16);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, rx1 + 4, labelY);
  }
}

interface Props {
  frame: FrameUpdate | null;
  isLiveActive: boolean;
}

export function LiveFeed({ frame, isLiveActive }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recvCountRef = useRef(0);
  const renderCountRef = useRef(0);
  const lastFrameNumberRef = useRef<number | null>(null);
  // Mutable mirror of `frame` so the 1s FPS timer can read the latest frame
  // without depending on `frame` (which would clear+recreate the interval
  // every WebSocket update, preventing the callback from ever firing).
  const frameRef = useRef<FrameUpdate | null>(frame);
  const [fpsStats, setFpsStats] = useState({ recv: 0, render: 0, ageMs: 0 });

  useEffect(() => {
    frameRef.current = frame;
    if (!frame) return;
    if (lastFrameNumberRef.current !== frame.frame_number) {
      lastFrameNumberRef.current = frame.frame_number;
      recvCountRef.current += 1;
    }
  }, [frame]);

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !frame) return;

    const render = () => {
      drawOverlay(canvas, img, frame.tracked_persons);
      renderCountRef.current += 1;
    };

    if (img.complete && img.naturalWidth > 0) {
      render();
    } else {
      img.onload = render;
    }
  }, [frame]);

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
  }, []);

  if (!isLiveActive) {
    return (
      <Card className="flex-1 flex items-center justify-center min-h-[400px]">
        <p className="text-muted-foreground text-sm">
          Live stream is paused. Press Start live to begin.
        </p>
      </Card>
    );
  }

  if (!frame) {
    return (
      <Card className="flex-1 flex items-center justify-center min-h-[400px]">
        <p className="text-muted-foreground text-sm">Waiting for frames…</p>
      </Card>
    );
  }

  return (
    <Card className="flex-1 relative overflow-hidden bg-black p-0 min-h-[400px]">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        ref={imgRef}
        src={`data:image/jpeg;base64,${frame.image_base64}`}
        alt="Live camera feed"
        className="w-full h-full object-contain"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{ objectFit: "contain" }}
      />
      <div className="absolute bottom-2 right-2 rounded bg-black/60 px-2 py-0.5 text-xs text-white/80">
        frame #{frame.frame_number}
      </div>
      <div className="absolute top-2 left-2 rounded bg-black/60 px-2 py-0.5 text-xs text-white/80">
        {frame.device_id}
        {frame.source ? ` • ${frame.source}` : ""}
      </div>
      <div className="absolute bottom-2 left-2 rounded bg-black/60 px-2 py-0.5 font-mono text-xs text-white/80">
        recv {fpsStats.recv} • render {fpsStats.render} • age {fpsStats.ageMs}ms
      </div>
    </Card>
  );
}
