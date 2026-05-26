import type { Tracklet, TrackletMatching, TrackletQuality } from "@/types";

type LiveMatching = TrackletMatching | null | undefined;
type LiveQuality =
  | {
      v_avg: number;
      embedding_consistency: number;
      overall_consistency: number;
      good_frame_ratio: number;
    }
  | null
  | undefined;

export function formatPct(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatDecimal(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function describeMatchMethod(matching: LiveMatching): string {
  switch (matching?.method) {
    case "gallery_match":
      return "Matched via gallery similarity";
    case "spatial_appearance_reuse":
      return "Recovered via spatial + appearance";
    case "recent_person_reuse":
      return "Reused recent identity";
    case "recent_person_reuse_after_fallback":
      return "Recovered after tentative fallback";
    case "tentative_soft_match":
      return "Recovered by soft match";
    case "current_identity_maintained":
      return "Maintained current identity";
    case "new_identity":
      if (matching.source === "tentative_promoted") return "Promoted from tentative track";
      if (matching.source === "tentative_fallback") return "Created after fallback";
      return "Created from strong tracklet evidence";
    case "ambiguous_rejected":
      return "Held due to ambiguous gallery match";
    case "tentative_pending":
      return "Waiting for enough clean evidence";
    default:
      return "Evidence pending";
  }
}

export function getLiveStatusLabel(status: string | null | undefined): string {
  switch (status) {
    case "recovering":
      return "Recovering";
    case "confirmed":
      return "Confirmed";
    default:
      return "Tentative";
  }
}

export function getStatusClasses(status: string | null | undefined): string {
  switch (status) {
    case "recovering":
      return "border-amber-300 bg-amber-50 text-amber-800";
    case "confirmed":
      return "border-emerald-300 bg-emerald-50 text-emerald-800";
    default:
      return "border-slate-300 bg-slate-100 text-slate-700";
  }
}

export function buildLiveEvidenceSummary(
  matching: LiveMatching,
  quality: LiveQuality,
  liveVisibilityScore: number,
  overlapRatio: number
): string {
  const parts: string[] = [describeMatchMethod(matching)];

  if (matching?.similarity_score !== null && matching?.similarity_score !== undefined) {
    parts.push(`sim ${matching.similarity_score.toFixed(2)}`);
  }

  if (quality?.good_frame_ratio !== null && quality?.good_frame_ratio !== undefined) {
    parts.push(`good ${formatPct(quality.good_frame_ratio)}`);
  }

  if (liveVisibilityScore < 0.45) {
    parts.push("current view occluded");
  } else if (overlapRatio >= 0.35) {
    parts.push("heavy overlap");
  }

  return parts.join(" • ");
}

export function summarizeTracklets(tracklets: Tracklet[]): {
  avgConsistency: number;
  avgGoodFrameRatio: number;
  recoveryCount: number;
  highQualityCount: number;
} {
  if (tracklets.length === 0) {
    return {
      avgConsistency: 0,
      avgGoodFrameRatio: 0,
      recoveryCount: 0,
      highQualityCount: 0,
    };
  }

  const avgConsistency =
    tracklets.reduce((sum, tracklet) => sum + (tracklet.quality.overall_consistency || 0), 0) /
    tracklets.length;
  const avgGoodFrameRatio =
    tracklets.reduce((sum, tracklet) => sum + (tracklet.quality.good_frame_ratio || 0), 0) /
    tracklets.length;
  const recoveryCount = tracklets.filter((tracklet) =>
    ["spatial_appearance_reuse", "recent_person_reuse", "recent_person_reuse_after_fallback", "tentative_soft_match"].includes(
      tracklet.matching.method
    )
  ).length;
  const highQualityCount = tracklets.filter(
    (tracklet) => tracklet.quality.overall_consistency >= 0.7
  ).length;

  return {
    avgConsistency,
    avgGoodFrameRatio,
    recoveryCount,
    highQualityCount,
  };
}

export function getTrackletConfidenceLabel(quality: TrackletQuality): string {
  if (quality.overall_consistency >= 0.8 && quality.good_frame_ratio >= 0.6) {
    return "High-confidence evidence";
  }
  if (quality.overall_consistency >= 0.65) {
    return "Moderate-confidence evidence";
  }
  return "Weak evidence";
}
