"use client";

import { useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import { getLiveStatusLabel } from "@/lib/reid-evidence";
import type { FrameUpdate, TrackedPerson } from "@/hooks/use-websocket";

function visColor(score: number): string {
  if (score >= 0.7) return "#22c55e";
  if (score >= 0.4) return "#eab308";
  return "#ef4444";
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
      status === "recovering"
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
}

export function LiveFeed({ frame }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !frame) return;

    const render = () => drawOverlay(canvas, img, frame.tracked_persons);

    if (img.complete && img.naturalWidth > 0) {
      render();
    } else {
      img.onload = render;
    }
  }, [frame]);

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
      </div>
    </Card>
  );
}
