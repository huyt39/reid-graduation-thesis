"use client";

import { Camera } from "lucide-react";

interface Props {
  deviceIds: string[];
  selected: string | null;
  onChange: (id: string) => void;
}

export default function DeviceSelector({ deviceIds, selected, onChange }: Props) {
  return (
    <div className="flex items-center gap-2">
      <Camera size={16} className="text-gray-400 shrink-0" />
      <select
        value={selected ?? ""}
        onChange={(e) => onChange(e.target.value)}
        className="bg-panel border border-border text-sm text-gray-200 rounded px-2 py-1 focus:outline-none focus:border-accent"
        disabled={deviceIds.length === 0}
      >
        {deviceIds.length === 0 ? (
          <option value="">No cameras</option>
        ) : (
          deviceIds.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))
        )}
      </select>
    </div>
  );
}
