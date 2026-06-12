"use client";

import { useState, useEffect } from "react";

// Module-level flag: stays true after the first client mount in the session,
// so navigations after the first load render the hydrated tree immediately
// (no placeholder flash on every route change).
let hydrated = false;

/**
 * Returns false during SSR and the first client (hydration) render, then flips
 * to true after mount. Use to gate client-only UI (e.g. Radix components whose
 * generated ids would otherwise cause a hydration mismatch) without a per-route
 * flash.
 */
export function useHydrated() {
  const [value, setValue] = useState(hydrated);

  useEffect(() => {
    hydrated = true;
    setValue(true);
  }, []);

  return value;
}
