"use client";

import { useState } from "react";
import Image from "next/image";
import { Expand, User } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

interface PersonSnapshotProps {
  src?: string | null;
  alt: string;
  label?: string;
  className?: string;
  imageClassName?: string;
  previewTitle?: string;
  previewDescription?: string;
}

export function PersonSnapshot({
  src,
  alt,
  label,
  className,
  imageClassName,
  previewTitle,
  previewDescription,
}: PersonSnapshotProps) {
  const [previewOpen, setPreviewOpen] = useState(false);

  if (!src) {
    return (
      <div className={cn("relative overflow-hidden rounded-lg border bg-muted/40", className)}>
        <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-muted-foreground">
          <User className="h-5 w-5" />
          {label ? <span className="text-[11px] font-medium">{label}</span> : null}
        </div>
      </div>
    );
  }

  return (
    <>
      <button
        type="button"
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          setPreviewOpen(true);
        }}
        className={cn(
          "group relative block overflow-hidden rounded-lg border bg-muted/40 text-left",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
          className
        )}
        aria-label={previewTitle ?? `Preview ${alt}`}
      >
        <Image
          src={src}
          alt={alt}
          fill
          sizes="(max-width: 768px) 40vw, 20vw"
          className={cn(
            "object-cover transition-transform duration-200 group-hover:scale-[1.02]",
            imageClassName
          )}
        />
        <div className="absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-black/70 via-black/20 to-transparent px-2 py-2 text-white opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100">
          <span className="text-[11px] font-medium">{label ?? "Preview"}</span>
          <Expand className="h-3.5 w-3.5" />
        </div>
      </button>

      <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
        <DialogContent className="max-h-[92vh] max-w-4xl overflow-hidden p-0 sm:max-w-5xl">
          <DialogHeader className="px-6 pt-6 pb-0">
            <DialogTitle>{previewTitle ?? alt}</DialogTitle>
            {previewDescription ? (
              <DialogDescription>{previewDescription}</DialogDescription>
            ) : null}
          </DialogHeader>
          <div className="px-6 pb-6 pt-4">
            <div className="relative overflow-hidden rounded-lg border bg-muted/30">
              <div className="relative aspect-[4/5] max-h-[72vh] min-h-80 w-full">
                <Image src={src} alt={alt} fill sizes="100vw" className="object-contain" />
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
