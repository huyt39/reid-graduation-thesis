import { loadToken, saveToken, clearToken, getRememberMe } from "@/lib/auth/token-storage";

type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
type QueryParamValue = string | number | boolean | undefined | null | Array<string | number>;

interface RequestConfig {
  method?: HttpMethod;
  headers?: Record<string, string>;
  body?: unknown;
  params?: Record<string, QueryParamValue>;
  skipAuth?: boolean;
}

interface ApiResponse<T> {
  data: T | null;
  error: string | null;
  status: number;
  errorData?: unknown;
}

/** Reid gateway URL. All /api/v1/* requests are proxied to the query service. */
const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:18080";

let isRefreshing = false;
let refreshSubscribers: ((token: string | null) => void)[] = [];

const notifySubscribers = (token: string | null) => {
  refreshSubscribers.forEach((cb) => cb(token));
  refreshSubscribers = [];
};

const waitForRefresh = (): Promise<string | null> =>
  new Promise((resolve) => {
    refreshSubscribers.push(resolve);
  });

class BaseApiClient {
  protected baseUrl: string;
  private defaultHeaders: Record<string, string>;

  constructor(baseUrl = `${GATEWAY_URL}/api/v1`) {
    this.baseUrl = baseUrl;
    this.defaultHeaders = { "Content-Type": "application/json" };
  }

  protected getAuthToken(): string | null {
    return loadToken().accessToken;
  }

  private buildUrl(endpoint: string, params?: Record<string, QueryParamValue>): string {
    const base = typeof window !== "undefined" ? window.location.origin : "http://localhost:3000";
    const url = new URL(`${this.baseUrl}${endpoint}`, base);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value === undefined || value === null || value === "") return;
        if (Array.isArray(value)) {
          value.forEach((v) => {
            if (v !== undefined && v !== null && v !== "") {
              url.searchParams.append(key, String(v));
            }
          });
        } else {
          url.searchParams.append(key, String(value));
        }
      });
    }
    return url.toString();
  }

  protected async refreshAccessToken(): Promise<string | null> {
    if (isRefreshing) return waitForRefresh();

    const { accessToken } = loadToken();
    if (!accessToken) return null;

    isRefreshing = true;
    try {
      const response = await fetch(`${GATEWAY_URL}/auth/refresh`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
      });
      if (response.ok) {
        const data = (await response.json()) as { access_token: string };
        const rememberMe = getRememberMe();
        saveToken(data.access_token, rememberMe);

        try {
          const { useAuthStore } = await import("@/lib/auth/auth-store");
          useAuthStore.getState().setAccessToken(data.access_token);
        } catch {
          // store not initialized yet
        }

        notifySubscribers(data.access_token);
        return data.access_token;
      }
      notifySubscribers(null);
      return null;
    } catch (error) {
      console.error("[BaseClient] Token refresh failed:", error);
      notifySubscribers(null);
      return null;
    } finally {
      isRefreshing = false;
    }
  }

  protected handleAuthFailure() {
    clearToken();
    if (typeof window !== "undefined") {
      const currentPath = window.location.pathname;
      if (currentPath !== "/sign-in") {
        window.location.href = `/sign-in?redirect=${encodeURIComponent(currentPath)}`;
      }
    }
  }

  async request<T>(endpoint: string, config: RequestConfig = {}): Promise<ApiResponse<T>> {
    const { method = "GET", headers = {}, body, params, skipAuth = false } = config;

    try {
      const token = skipAuth ? null : this.getAuthToken();
      const requestHeaders = {
        ...this.defaultHeaders,
        ...headers,
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      };

      const url = this.buildUrl(endpoint, params);

      let response = await fetch(url, {
        method,
        headers: requestHeaders,
        body: body ? JSON.stringify(body) : undefined,
      });

      if (response.status === 401 && !skipAuth) {
        const newToken = await this.refreshAccessToken();
        if (newToken) {
          response = await fetch(url, {
            method,
            headers: {
              ...this.defaultHeaders,
              ...headers,
              Authorization: `Bearer ${newToken}`,
            },
            body: body ? JSON.stringify(body) : undefined,
          });
        } else {
          this.handleAuthFailure();
          return { data: null, error: "Session expired", status: 401 };
        }
      }

      const responseData = await response.json().catch(() => ({}));

      if (!response.ok) {
        const err = responseData as { detail?: string; message?: string; error?: string };
        return {
          data: null,
          error: err.detail || err.message || err.error || "Request failed",
          status: response.status,
          errorData: responseData,
        };
      }

      return { data: responseData as T, error: null, status: response.status };
    } catch (error) {
      console.error("[BaseClient] Request failed:", error);
      return {
        data: null,
        error: error instanceof Error ? error.message : "Network error",
        status: 500,
      };
    }
  }

  async get<T>(
    endpoint: string,
    params?: Record<string, QueryParamValue>,
    skipAuth?: boolean
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: "GET", params, skipAuth });
  }

  async post<T>(endpoint: string, body?: unknown, skipAuth?: boolean): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: "POST", body, skipAuth });
  }

  async put<T>(endpoint: string, body?: unknown, skipAuth?: boolean): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: "PUT", body, skipAuth });
  }

  async patch<T>(endpoint: string, body?: unknown, skipAuth?: boolean): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: "PATCH", body, skipAuth });
  }

  async delete<T>(endpoint: string, skipAuth?: boolean): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: "DELETE", skipAuth });
  }
}

export default BaseApiClient;
export type { ApiResponse, RequestConfig };
export const apiClient = new BaseApiClient();
