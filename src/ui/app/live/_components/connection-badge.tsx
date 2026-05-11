"use client";

import { Wifi, WifiOff, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { ConnectionState } from "@/hooks/use-websocket";

export function ConnectionBadge({ state }: { state: ConnectionState }) {
  if (state === "connected") {
    return (
      <Badge className="bg-emerald-600 hover:bg-emerald-600">
        <Wifi className="h-3 w-3" />
        Live
      </Badge>
    );
  }
  if (state === "connecting") {
    return (
      <Badge variant="secondary">
        <Loader2 className="h-3 w-3 animate-spin" />
        Connecting
      </Badge>
    );
  }
  return (
    <Badge variant="destructive">
      <WifiOff className="h-3 w-3" />
      Disconnected
    </Badge>
  );
}
