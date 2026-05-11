import BaseApiClient, { type ApiResponse } from "./base-client";
import type { NLQueryResult } from "@/types";

class SearchClient extends BaseApiClient {
  structured(queryType: string, params: Record<string, unknown>): Promise<ApiResponse<unknown>> {
    return this.post<unknown>("/search", { query_type: queryType, params });
  }

  natural(query: string): Promise<ApiResponse<NLQueryResult>> {
    return this.post<NLQueryResult>("/query/natural", { query });
  }
}

export const searchClient = new SearchClient();
