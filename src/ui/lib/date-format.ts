import { format, formatDistanceToNow, parseISO } from "date-fns";

function parseApiDate(value: string | Date): Date {
  if (value instanceof Date) return value;

  const normalized = value.trim();
  const hasExplicitTimezone = normalized.endsWith("Z") || /[+-]\d{2}:\d{2}$/.test(normalized);

  return parseISO(hasExplicitTimezone ? normalized : `${normalized}Z`);
}

export function formatDateTime(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const date = parseApiDate(value);
  return format(date, "yyyy-MM-dd HH:mm:ss");
}

export function formatDate(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const date = parseApiDate(value);
  return format(date, "yyyy-MM-dd");
}

export function formatRelative(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const date = parseApiDate(value);
  return formatDistanceToNow(date, { addSuffix: true });
}
