import BaseApiClient, { type ApiResponse } from "./base-client";
import type { Stats, AggregationResponse } from "@/types";

class StatsClient extends BaseApiClient {
  getStats(): Promise<ApiResponse<Stats>> {
    return this.get<Stats>("/stats");
  }

  aggregate(
    params: {
      person_id?: number;
      device_id?: string;
      start_time?: string;
      end_time?: string;
      group_by?: "hour" | "day" | "device";
    } = {}
  ): Promise<ApiResponse<AggregationResponse>> {
    return this.get<AggregationResponse>("/stats/aggregate", params);
  }
}

export const statsClient = new StatsClient();
