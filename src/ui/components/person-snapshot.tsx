"use client";

import Image from "next/image";
import { User } from "lucide-react";
import { cn } from "@/lib/utils";

interface PersonSnapshotProps {
  src?: string | null;
  alt: string;
  label?: string;
  className?: string;
  imageClassName?: string;
}

export function PersonSnapshot({
  src,
  alt,
  label,
  className,
  imageClassName,
}: PersonSnapshotProps) {
  return (
    <div className={cn("relative overflow-hidden rounded-lg border bg-muted/40", className)}>
      {src ? (
        <Image
          src={src}
          alt={alt}
          fill
          sizes="(max-width: 768px) 40vw, 20vw"
          className={cn("object-cover", imageClassName)}
        />
      ) : (
        <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-muted-foreground">
          <User className="h-5 w-5" />
          {label ? <span className="text-[11px] font-medium">{label}</span> : null}
        </div>
      )}
    </div>
  );
}
