/**
 * Auth API Client
 *
 * Talks to the reid gateway:
 *   POST /auth/login    { username, password } -> { access_token, ... }
 *   POST /auth/refresh  (Bearer)               -> { access_token, ... }
 *
 * There is no /auth/me endpoint; the user (username + role) is decoded from
 * the JWT payload.
 */
import type { ApiResponse } from "../api/base-client";
import type { SignInFormData, TokenResponse, User, Role } from "./types";
import { loadToken, saveToken, clearToken, getRememberMe } from "./token-storage";

const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:18080";

let isRefreshing = false;
let refreshPromise: Promise<string | null> | null = null;

interface JwtPayload {
  sub: string;
  role: Role;
  exp: number;
}

export function decodeJwt(token: string): JwtPayload | null {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    const raw = atob(parts[1].replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(raw) as JwtPayload;
  } catch {
    return null;
  }
}

export function userFromToken(token: string): User | null {
  const payload = decodeJwt(token);
  if (!payload) return null;
  return { username: payload.sub, role: payload.role };
}

class AuthApiClient {
  private baseUrl: string;

  constructor(baseUrl = GATEWAY_URL) {
    this.baseUrl = baseUrl;
  }

  private async json<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          ...options.headers,
        },
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        return {
          data: null,
          error:
            (data as { detail?: string; message?: string }).detail ||
            (data as { message?: string }).message ||
            "Request failed",
          status: response.status,
        };
      }

      return { data: data as T, error: null, status: response.status };
    } catch (error) {
      return {
        data: null,
        error: error instanceof Error ? error.message : "Network error",
        status: 500,
      };
    }
  }

  async signIn(credentials: SignInFormData): Promise<ApiResponse<TokenResponse>> {
    return this.json<TokenResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify(credentials),
    });
  }

  /**
   * Refresh by calling /auth/refresh with the current access token.
   * Uses a singleton promise to dedupe concurrent refresh attempts.
   */
  async refreshAccessToken(): Promise<string | null> {
    if (isRefreshing && refreshPromise) return refreshPromise;

    const { accessToken } = loadToken();
    if (!accessToken) return null;

    isRefreshing = true;
    refreshPromise = this.performRefresh(accessToken);

    try {
      return await refreshPromise;
    } finally {
      isRefreshing = false;
      refreshPromise = null;
    }
  }

  private async performRefresh(currentToken: string): Promise<string | null> {
    const response = await this.json<TokenResponse>("/auth/refresh", {
      method: "POST",
      headers: { Authorization: `Bearer ${currentToken}` },
    });

    if (!response.data) return null;

    const { access_token } = response.data;
    const rememberMe = getRememberMe();
    saveToken(access_token, rememberMe);

    try {
      const { useAuthStore } = await import("./auth-store");
      useAuthStore.getState().setAccessToken(access_token);
    } catch {
      // Store may not be initialized yet — token is already in storage.
    }

    return access_token;
  }

  async signOut(): Promise<void> {
    clearToken();
  }
}

export const authClient = new AuthApiClient();
export default AuthApiClient;
