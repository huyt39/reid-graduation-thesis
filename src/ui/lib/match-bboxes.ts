import type { TrackedPerson } from "@/hooks/use-websocket";

export interface LiveIdentityCacheEntry {
  person: TrackedPerson;
  frameNumber: number;
  updatedAt: number;
}

interface MergeOptions {
  minIou?: number;
  cachedPersons?: TrackedPerson[];
  sourceSize?: { width: number | null; height: number | null };
  targetSize?: { width: number | null; height: number | null };
}

/**
 * IoU between two [x1, y1, x2, y2] boxes.
 */
function iou(a: [number, number, number, number], b: [number, number, number, number]): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

function scaleBbox(
  bbox: [number, number, number, number],
  sourceSize?: { width: number | null; height: number | null },
  targetSize?: { width: number | null; height: number | null }
): [number, number, number, number] {
  const sw = sourceSize?.width ?? null;
  const sh = sourceSize?.height ?? null;
  const tw = targetSize?.width ?? null;
  const th = targetSize?.height ?? null;
  if (!sw || !sh || !tw || !th || sw <= 0 || sh <= 0 || tw <= 0 || th <= 0) {
    return bbox;
  }
  if (sw === tw && sh === th) return bbox;
  const sx = tw / sw;
  const sy = th / sh;
  return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy];
}

/**
 * Project person_ids + attributes from the latest processed frame onto
 * the current raw frame's bboxes via greedy IoU matching.
 *
 * Returns a new TrackedPerson list with the same bboxes as `rawPersons`
 * but enriched with `person_id` + attribute fields from `processedPersons`
 * wherever a match above `minIou` exists. Unmatched raw bboxes keep
 * `person_id: null` so the overlay falls back to a track-only label.
 */
export function mergeRawWithProcessedIds(
  rawPersons: TrackedPerson[],
  processedPersons: TrackedPerson[],
  options: MergeOptions | number = {}
): TrackedPerson[] {
  const minIou = typeof options === "number" ? options : (options.minIou ?? 0.4);
  const cachedPersons = typeof options === "number" ? [] : (options.cachedPersons ?? []);
  const sourceSize = typeof options === "number" ? undefined : options.sourceSize;
  const targetSize = typeof options === "number" ? undefined : options.targetSize;
  const candidates = [...processedPersons, ...cachedPersons];
  if (!candidates.length) return rawPersons;

  const matchedProcessed = new Set<number>();
  const processedByTracklet = new Map<string, number>();

  candidates.forEach((person, index) => {
    if (person.tracklet_id) {
      processedByTracklet.set(person.tracklet_id, index);
    }
  });

  return rawPersons.map((rawP) => {
    let bestIdx = -1;
    if (rawP.tracklet_id) {
      const trackletMatch = processedByTracklet.get(rawP.tracklet_id);
      if (trackletMatch !== undefined && !matchedProcessed.has(trackletMatch)) {
        bestIdx = trackletMatch;
      }
    }

    if (bestIdx < 0) {
      let bestIou = minIou;
      for (let i = 0; i < candidates.length; i++) {
        if (matchedProcessed.has(i)) continue;
        const procP = candidates[i];
        if (procP.person_id == null) continue;
        const procBbox = scaleBbox(procP.bbox, sourceSize, targetSize);
        const score = iou(rawP.bbox, procBbox);
        if (score > bestIou) {
          bestIou = score;
          bestIdx = i;
        }
      }
    }

    if (bestIdx < 0) return rawP;
    matchedProcessed.add(bestIdx);
    const proc = candidates[bestIdx];
    return {
      ...rawP,
      ...proc,
      bbox: rawP.bbox,
      confidence: rawP.confidence,
      live_visibility_score: rawP.live_visibility_score,
      overlap_ratio: rawP.overlap_ratio,
    };
  });
}

export function updateLiveIdentityCache(
  cache: Map<string, LiveIdentityCacheEntry>,
  persons: TrackedPerson[],
  frameNumber: number,
  now = Date.now()
): void {
  for (const person of persons) {
    if (person.person_id == null || person.tracklet_state === "tentative") continue;
    const key = person.live_track_key ?? person.tracklet_id ?? `person:${person.person_id}`;
    cache.set(key, {
      person,
      frameNumber,
      updatedAt: now,
    });
  }
}

export function getCachedLiveIdentities(
  cache: Map<string, LiveIdentityCacheEntry>,
  frameNumber: number,
  now = Date.now(),
  maxFrameLag = 120,
  maxAgeMs = 5000
): TrackedPerson[] {
  const persons: TrackedPerson[] = [];
  for (const [key, entry] of cache) {
    const tooOldByFrame = frameNumber - entry.frameNumber > maxFrameLag;
    const tooOldByTime = now - entry.updatedAt > maxAgeMs;
    if (tooOldByFrame || tooOldByTime) {
      cache.delete(key);
      continue;
    }
    persons.push(entry.person);
  }
  return persons;
}
