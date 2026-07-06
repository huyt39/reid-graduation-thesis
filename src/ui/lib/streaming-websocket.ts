const DEFAULT_STREAMING_WS_URL = "ws://localhost:18080/ws/stream";

export function getStreamingWebSocketUrl(): string {
  const configuredUrl = process.env.NEXT_PUBLIC_STREAMING_WS || DEFAULT_STREAMING_WS_URL;
  const normalizedUrl = configuredUrl.replace(/\/+$/, "");

  if (normalizedUrl.endsWith("/ws") || normalizedUrl.endsWith("/ws/stream")) {
    return normalizedUrl;
  }

  return `${normalizedUrl}/ws`;
}
