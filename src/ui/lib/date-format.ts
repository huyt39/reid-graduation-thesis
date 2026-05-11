import { format, formatDistanceToNow, parseISO } from "date-fns";

export function formatDateTime(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const date = typeof value === "string" ? parseISO(value) : value;
  return format(date, "yyyy-MM-dd HH:mm:ss");
}

export function formatDate(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const date = typeof value === "string" ? parseISO(value) : value;
  return format(date, "yyyy-MM-dd");
}

export function formatRelative(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const date = typeof value === "string" ? parseISO(value) : value;
  return formatDistanceToNow(date, { addSuffix: true });
}
