const TOKEN_KEY = "reid_token";
const DEFAULT_GATEWAY_URL = "http://localhost:18080";
const DEV_USERNAME = "admin";
const DEV_PASSWORD = "admin";
let autoLoginPromise: Promise<void> | null = null;

export interface TokenPayload {
  sub: string;
  role: string;
  exp: number;
}

function decodePayload(token: string): TokenPayload | null {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    const raw = atob(parts[1].replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(raw) as TokenPayload;
  } catch {
    return null;
  }
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

export function isAuthenticated(): boolean {
  const token = getToken();
  if (!token) return false;
  const payload = decodePayload(token);
  if (!payload) return false;
  // exp is Unix seconds
  return payload.exp * 1000 > Date.now();
}

export function getUser(): TokenPayload | null {
  const token = getToken();
  if (!token) return null;
  return decodePayload(token);
}

export async function login(username: string, password: string): Promise<void> {
  const gatewayUrl = process.env.NEXT_PUBLIC_GATEWAY_URL ?? DEFAULT_GATEWAY_URL;
  const res = await fetch(`${gatewayUrl}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? "Login failed");
  }
  const data = (await res.json()) as { access_token: string };
  setToken(data.access_token);
}

export async function ensureDevToken(): Promise<string | null> {
  const existing = getToken();
  if (existing) return existing;

  if (!autoLoginPromise) {
    autoLoginPromise = login(DEV_USERNAME, DEV_PASSWORD).finally(() => {
      autoLoginPromise = null;
    });
  }

  try {
    await autoLoginPromise;
  } catch {
    return null;
  }

  return getToken();
}

export function logout(): void {
  clearToken();
  window.location.href = "/dashboard";
}
