"use client";

import { Camera } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// Sentinel value for the "show every camera side-by-side" option. Maps to a
// null selectedDevice in the parent (Radix Select can't use null/"" as a value).
export const ALL_CAMERAS = "__all__";

interface Props {
  deviceIds: string[];
  selected: string | null;
  onChange: (id: string) => void;
}

export function DeviceSelector({ deviceIds, selected, onChange }: Props) {
  const empty = deviceIds.length === 0;
  return (
    <div className="flex items-center gap-2">
      <Camera className="h-4 w-4 text-muted-foreground shrink-0" />
      <Select
        value={selected ?? ALL_CAMERAS}
        onValueChange={onChange}
        disabled={empty}
      >
        <SelectTrigger className="w-56">
          <SelectValue placeholder={empty ? "No cameras" : "Select camera"} />
        </SelectTrigger>
        <SelectContent>
          {deviceIds.length > 1 ? (
            <SelectItem value={ALL_CAMERAS}>{deviceIds.length} cameras</SelectItem>
          ) : null}
          {deviceIds.map((id) => (
            <SelectItem key={id} value={id}>
              {id}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
