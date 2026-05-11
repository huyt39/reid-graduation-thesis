"use client";

import { Camera } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

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
        value={selected ?? undefined}
        onValueChange={onChange}
        disabled={empty}
      >
        <SelectTrigger className="w-56">
          <SelectValue placeholder={empty ? "No cameras" : "Select camera"} />
        </SelectTrigger>
        <SelectContent>
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
