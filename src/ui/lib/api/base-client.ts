type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
type QueryParamValue = string | number | boolean | undefined | null | Array<string | number>;

interface RequestConfig {
  method?: HttpMethod;
  headers?: Record<string, string>;
  body?: unknown;
  params?: Record<string, QueryParamValue>;
}

interface ApiResponse<T> {
  data: T | null;
  error: string | null;
  status: number;
  errorData?: unknown;
}

/** Reid gateway URL. All /api/v1/* requests are proxied to the query service. */
const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:18080";

class BaseApiClient {
  protected baseUrl: string;
  private defaultHeaders: Record<string, string>;

  constructor(baseUrl = `${GATEWAY_URL}/api/v1`) {
    this.baseUrl = baseUrl;
    this.defaultHeaders = { "Content-Type": "application/json" };
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

  async request<T>(endpoint: string, config: RequestConfig = {}): Promise<ApiResponse<T>> {
    const { method = "GET", headers = {}, body, params } = config;

    try {
      const requestHeaders = {
        ...this.defaultHeaders,
        ...headers,
        ...(method === "GET"
          ? {
              "Cache-Control": "no-cache, no-store, must-revalidate",
              Pragma: "no-cache",
            }
          : {}),
      };

      const url = this.buildUrl(endpoint, params);

      const response = await fetch(url, {
        method,
        headers: requestHeaders,
        body: body ? JSON.stringify(body) : undefined,
        cache: method === "GET" ? "no-store" : "default",
      });

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
    params?: Record<string, QueryParamValue>
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: "GET", params });
  }

  async post<T>(endpoint: string, body?: unknown): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: "POST", body });
  }

  async put<T>(endpoint: string, body?: unknown): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: "PUT", body });
  }

  async patch<T>(endpoint: string, body?: unknown): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: "PATCH", body });
  }

  async delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: "DELETE" });
  }
}

export default BaseApiClient;
export type { ApiResponse, RequestConfig };
export const apiClient = new BaseApiClient();
