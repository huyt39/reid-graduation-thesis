"use client";

import { Card } from "@/components/ui/card";

// The Live tab is a plain monitoring view. The raw video is served as MJPEG by
// the standalone raw_stream service (decoupled from ReID), rendered natively by
// the browser <img> for smoothness. Person identities/attributes/evidence live
// in the Persons/Timeline tabs, so no per-frame bbox overlay is drawn here.
interface Props {
  deviceId: string | null;
  mjpegUrl: string | null;
  isLiveActive: boolean;
}

export function LiveFeed({ deviceId, mjpegUrl, isLiveActive }: Props) {
  if (!isLiveActive) {
    return (
      <Card className="flex-1 flex items-center justify-center aspect-video">
        <p className="text-muted-foreground text-sm">
          Live stream is paused. Press Start live to begin.
        </p>
      </Card>
    );
  }

  if (!mjpegUrl) {
    return (
      <Card className="flex-1 flex items-center justify-center aspect-video">
        <p className="text-muted-foreground text-sm">Waiting for stream…</p>
      </Card>
    );
  }

  return (
    <Card className="flex-1 relative overflow-hidden bg-black p-0 aspect-video">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={mjpegUrl}
        alt="Live camera feed"
        className="w-full h-full object-contain"
      />
      {deviceId ? (
        <div className="absolute top-2 left-2 rounded bg-black/60 px-2 py-0.5 text-xs text-white/80">
          {deviceId}
        </div>
      ) : null}
    </Card>
  );
}
