"use client";

import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

// The Live tab is a plain monitoring view. The raw video is served as MJPEG by
// the standalone raw_stream service (decoupled from ReID), rendered natively by
// the browser <img> for smoothness. Person identities/attributes/evidence live
// in the Persons/Timeline tabs, so no per-frame bbox overlay is drawn here.
//
// Pause behavior: instead of tearing the video down, we snapshot the last
// rendered frame into a hidden <canvas> and show that frozen image while paused
// (so the user still sees where the stream stopped), then drop the live <img>
// to close the MJPEG connection — which lets the lazy backend reader idle.
// raw_stream sends CORS `*`, so the <img crossOrigin="anonymous"> -> canvas
// readback is not tainted and toDataURL() works.
interface Props {
  deviceId: string | null;
  mjpegUrl: string | null;
  isLiveActive: boolean;
}

export function LiveFeed({ deviceId, mjpegUrl, isLiveActive }: Props) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [frozen, setFrozen] = useState<string | null>(null);
  // The live <img> stays mounted while streaming; we only unmount it AFTER a
  // pause snapshot has been captured, so the ref is still valid at capture time.
  const [streaming, setStreaming] = useState(isLiveActive);

  useEffect(() => {
    if (isLiveActive) {
      setFrozen(null);
      setStreaming(true);
      return;
    }
    // Paused: capture the currently-displayed frame while <img> is still mounted.
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (img && canvas && img.naturalWidth > 0) {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        try {
          ctx.drawImage(img, 0, 0);
          setFrozen(canvas.toDataURL("image/jpeg"));
        } catch {
          // Readback failed (unexpected with CORS *) — fall back to a text badge.
          setFrozen(null);
        }
      }
    }
    setStreaming(false);
  }, [isLiveActive]);

  if (!mjpegUrl) {
    return (
      <Card className="flex-1 flex items-center justify-center aspect-video">
        <p className="text-muted-foreground text-sm">Waiting for stream…</p>
      </Card>
    );
  }

  return (
    <Card className="flex-1 relative overflow-hidden bg-black p-0 aspect-video">
      {/* Hidden scratch canvas used only to snapshot the last frame on pause. */}
      <canvas ref={canvasRef} className="hidden" />

      {streaming ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          ref={imgRef}
          src={mjpegUrl}
          crossOrigin="anonymous"
          alt="Live camera feed"
          className="w-full h-full object-contain"
        />
      ) : frozen ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={frozen}
          alt="Paused camera feed"
          className="w-full h-full object-contain"
        />
      ) : (
        <div className="w-full h-full flex items-center justify-center">
          <p className="text-muted-foreground text-sm">Paused</p>
        </div>
      )}

      {!isLiveActive ? (
        <div className="absolute top-2 right-2">
          <Badge variant="outline" className="bg-black/60 text-white/90 border-white/30">
            Paused
          </Badge>
        </div>
      ) : null}
      {deviceId ? (
        <div className="absolute top-2 left-2 rounded bg-black/60 px-2 py-0.5 text-xs text-white/80">
          {deviceId}
        </div>
      ) : null}
    </Card>
  );
}
