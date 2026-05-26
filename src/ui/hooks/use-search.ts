"use client";

import { useState, useCallback } from "react";
import { searchClient } from "@/lib/api/search-client";

interface UseSearchResult {
  result: unknown;
  isLoading: boolean;
  error: string | null;
  run: (queryType: string, params: Record<string, unknown>) => Promise<void>;
  reset: () => void;
}

export function useSearch(): UseSearchResult {
  const [result, setResult] = useState<unknown>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (queryType: string, params: Record<string, unknown>) => {
    setIsLoading(true);
    setError(null);
    const response = await searchClient.structured(queryType, params);
    if (response.error) {
      setError(response.error);
      setResult(null);
    } else {
      setResult(response.data);
    }
    setIsLoading(false);
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, isLoading, error, run, reset };
}
