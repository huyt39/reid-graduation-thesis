/**
 * Shared Token Storage Helpers
 *
 * Single source of truth for token persistence logic.
 * Used by auth-store.ts, auth-client.ts, and base-client.ts.
 *
 * Strategy:
 * - rememberMe = true  → token in localStorage (survives browser close)
 * - rememberMe = false → token in sessionStorage (cleared on browser close)
 *
 * The reid gateway issues a single access token (no separate refresh token);
 * /auth/refresh re-issues a new token from a valid bearer token.
 */

export const ACCESS_TOKEN_KEY = "reid_access_token";
export const REMEMBER_ME_KEY = "reid_remember_me";

export const saveToken = (accessToken: string | null, rememberMe: boolean) => {
  if (typeof window === "undefined") return;

  const storage = rememberMe ? localStorage : sessionStorage;
  const otherStorage = rememberMe ? sessionStorage : localStorage;

  otherStorage.removeItem(ACCESS_TOKEN_KEY);

  if (accessToken) {
    storage.setItem(ACCESS_TOKEN_KEY, accessToken);
  } else {
    storage.removeItem(ACCESS_TOKEN_KEY);
  }

  localStorage.setItem(REMEMBER_ME_KEY, String(rememberMe));
};

export const loadToken = (): { accessToken: string | null } => {
  if (typeof window === "undefined") return { accessToken: null };

  let accessToken = localStorage.getItem(ACCESS_TOKEN_KEY);
  if (!accessToken) {
    accessToken = sessionStorage.getItem(ACCESS_TOKEN_KEY);
  }
  return { accessToken };
};

export const clearToken = () => {
  if (typeof window === "undefined") return;
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  sessionStorage.removeItem(ACCESS_TOKEN_KEY);
};

export const loadRememberMe = (): boolean => {
  if (typeof window === "undefined") return false;
  return localStorage.getItem(REMEMBER_ME_KEY) === "true";
};

export const getRememberMe = (): boolean => loadRememberMe();
