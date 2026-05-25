import type { TrackedPerson } from "@/hooks/use-websocket";

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
  minIou = 0.4
): TrackedPerson[] {
  if (!processedPersons.length) return rawPersons;

  const matchedProcessed = new Set<number>();
  const processedByTracklet = new Map<string, number>();

  processedPersons.forEach((person, index) => {
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
      for (let i = 0; i < processedPersons.length; i++) {
        if (matchedProcessed.has(i)) continue;
        const procP = processedPersons[i];
        if (procP.person_id == null) continue;
        const score = iou(rawP.bbox, procP.bbox);
        if (score > bestIou) {
          bestIou = score;
          bestIdx = i;
        }
      }
    }

    if (bestIdx < 0) return rawP;
    matchedProcessed.add(bestIdx);
    const proc = processedPersons[bestIdx];
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
