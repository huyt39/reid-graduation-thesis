"use client";

import { useEffect, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import type { FrameUpdate, TrackedPerson } from "@/hooks/use-websocket";

// The Live tab is a plain monitoring view (raw edge video + person-detection
// boxes). ReID identity/colour/score belongs to the Persons/Timeline tabs, not
// here — so every detected person is drawn as one neutral box, no label.
const BOX_COLOR = "#22c55e";

function drawOverlay(canvas: HTMLCanvasElement, img: HTMLImageElement, persons: TrackedPerson[]) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const W = canvas.width;
  const H = canvas.height;

  ctx.strokeStyle = BOX_COLOR;
  ctx.lineWidth = 2;
  for (const p of persons) {
    const [x1, y1, x2, y2] = p.bbox;
    const isNorm = x2 <= 1.0 && y2 <= 1.0;
    const rx1 = isNorm ? x1 * W : x1;
    const ry1 = isNorm ? y1 * H : y1;
    const rx2 = isNorm ? x2 * W : x2;
    const ry2 = isNorm ? y2 * H : y2;
    ctx.strokeRect(rx1, ry1, rx2 - rx1, ry2 - ry1);
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
      <Card className="flex-1 flex items-center justify-center aspect-video">
        <p className="text-muted-foreground text-sm">
          Live stream is paused. Press Start live to begin.
        </p>
      </Card>
    );
  }

  if (!frame) {
    return (
      <Card className="flex-1 flex items-center justify-center aspect-video">
        <p className="text-muted-foreground text-sm">Waiting for frames…</p>
      </Card>
    );
  }

  return (
    <Card className="flex-1 relative overflow-hidden bg-black p-0 aspect-video">
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
