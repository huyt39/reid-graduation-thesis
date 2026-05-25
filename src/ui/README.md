# reid-production-ui

ReID dashboard UI. Next.js 16 (App Router) + React 19 + Tailwind v4 + shadcn/ui.

Architecture mirrors the Cawinpod-fe pattern: feature folders under `app/` with
private `_components/` / `_hooks/`, shadcn primitives in `components/ui/`,
SWR + Zustand for data + auth, and a `BaseApiClient` against the gateway.

## Develop

Requires **Node.js ≥ 20.9** (Next.js 16 requirement). Tested on Node 22.

```bash
pnpm install
pnpm dev   # http://localhost:3000
```

The gateway must be reachable at `NEXT_PUBLIC_GATEWAY_URL`
(default `http://localhost:18080`) and the streaming WebSocket at
`NEXT_PUBLIC_STREAMING_WS`.

## Folder layout

- `app/` — App Router routes (`sign-in`, `dashboard`, `persons`, `devices`,
  `search`, `timeline`). Page-private files in `_components/`, `_hooks/`.
- `components/ui/` — shadcn primitives.
- `components/auth/` — `AuthGuard`, `SignInForm`.
- `components/dashboard-layout.tsx` — sidebar + header shell.
- `lib/auth/` — Zustand store, auth API client, token storage.
- `lib/api/` — `BaseApiClient` and per-feature clients against `/api/v1/*`.
- `hooks/` — SWR wrappers and the streaming WebSocket hook.
- `types/` — domain types mirrored from the query-service schemas.
