"use client";

import { useState, useCallback } from "react";
import { searchClient } from "@/lib/api/search-client";
import type { NLQueryResult } from "@/types";

interface UseSearchResult {
  result: NLQueryResult | null;
  isLoading: boolean;
  error: string | null;
  runNatural: (query: string) => Promise<void>;
  reset: () => void;
}

export function useSearch(): UseSearchResult {
  const [result, setResult] = useState<NLQueryResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runNatural = useCallback(async (query: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await searchClient.natural(query);
      if (response.error) {
        setError(response.error);
        setResult(null);
      } else {
        setResult(response.data);
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, isLoading, error, runNatural, reset };
}
