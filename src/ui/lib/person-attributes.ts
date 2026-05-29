import type { PersonAttributes } from "@/types";

export type AttributeTone = "default" | "muted";

export interface AttributeDisplayItem {
  key: string;
  label: string;
  value: string;
  confidence: number | null;
  tone: AttributeTone;
}

export interface AttributeDisplayGroup {
  title: string;
  items: AttributeDisplayItem[];
}

const GROUP_ORDER = ["Profile", "Accessories", "Clothing", "Other"] as const;

const KNOWN_ATTRIBUTE_META: Record<string, { label: string; group: (typeof GROUP_ORDER)[number] }> = {
  gender: { label: "Gender", group: "Profile" },
  age_child: { label: "Age", group: "Profile" },
  backpack: { label: "Backpack", group: "Accessories" },
  sidebag: { label: "Side bag", group: "Accessories" },
  hat: { label: "Hat", group: "Accessories" },
  glasses: { label: "Glasses", group: "Accessories" },
  sleeve: { label: "Sleeve", group: "Clothing" },
  lower: { label: "Lower", group: "Clothing" },
};

// Attributes whose classifier output is currently too unreliable to
// surface in the UI. Data still flows through worker → Kafka → DB; the
// fields are simply filtered out at render time. Remove from this set
// to re-enable display once the underlying signal is fixed.
const HIDDEN_ATTRIBUTE_KEYS = new Set(["glasses"]);

function normalizeLabel(value: string): string {
  return value.replace(/_/g, " ");
}

function normalizeKey(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function toNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function confidenceLabel(confidence: number | null): string {
  if (confidence === null) return "—";
  return `${Math.round(confidence * 100)}%`;
}

function getAttributeItems(attributes: PersonAttributes): AttributeDisplayItem[] {
  return Object.entries(attributes)
    .filter(([key]) => !key.endsWith("_confidence"))
    .filter(([key]) => !HIDDEN_ATTRIBUTE_KEYS.has(key))
    .flatMap(([key, rawValue]) => {
      if (rawValue === undefined || rawValue === null) return [];
      const value = String(rawValue);
      if (!value || value === "unknown") return [];
      const confidence = toNumber(attributes[`${key}_confidence`]);
      const tone: AttributeTone = confidence !== null && confidence >= 0.85 ? "default" : "muted";
      const meta = KNOWN_ATTRIBUTE_META[key];

      return [
        {
          key,
          label: meta?.label ?? normalizeKey(key),
          value: normalizeLabel(value),
          confidence,
          tone,
        },
      ];
    })
    .sort((a, b) => {
      const aKnown = KNOWN_ATTRIBUTE_META[a.key] ? 0 : 1;
      const bKnown = KNOWN_ATTRIBUTE_META[b.key] ? 0 : 1;
      if (aKnown !== bKnown) return aKnown - bKnown;
      return a.label.localeCompare(b.label);
    });
}

export function getAttributeGroups(attributes: PersonAttributes): AttributeDisplayGroup[] {
  const items = getAttributeItems(attributes);
  return GROUP_ORDER.map((title) => ({
    title,
    items: items.filter((item) => (KNOWN_ATTRIBUTE_META[item.key]?.group ?? "Other") === title),
  })).filter((group) => group.items.length > 0);
}

export function getCompactAttributes(
  attributes: PersonAttributes,
  maxItems = Number.POSITIVE_INFINITY
): AttributeDisplayItem[] {
  const items = getAttributeGroups(attributes).flatMap((group) => group.items);
  return items
    .filter((item) => item.key !== "gender")
    .sort((a, b) => (b.confidence ?? -1) - (a.confidence ?? -1))
    .slice(0, maxItems);
}
