# iNOB web ↔ backend API contract

The Next.js app (`gui/web`) talks to the FastAPI backend (`gui/backend/app.py`).
HTTP `/api/*` is proxied to `http://127.0.0.1:8000` by Next rewrites
(`next.config.mjs`); the run WebSocket is connected to directly at
`ws://127.0.0.1:8000/api/run` (override with `NEXT_PUBLIC_INOB_WS`).

## Already implemented by the backend (used as-is)

| Method | Path | Use |
|--------|------|-----|
| GET | `/api/health` | online indicator |
| GET | `/api/config` | `{config: {...}, errors: []}` — nested dict |
| PUT | `/api/config` | body `{config}` → `{ok, errors}` (validated by `inob.config.load_config`) |
| POST | `/api/config/validate` | body `{config}` → `{errors: []}` |
| GET | `/api/meshes` | `{meshes: [{name, url, bytes}]}` |
| GET | `/api/meshes/{name}` | streams an STL |
| WS | `/api/run` | client→`{stages, force, ...}`; server→`{type:"log",line}` … `{type:"done",statuses}` / `{type:"error",message}` |

## New contract the backend should implement (frontend already speaks it)

### 1. Run init payload — additional fields
The client sends, on WS open:
```json
{ "stages": ["geom","fem","sensors","forward","viz"],
  "force": false,
  "sources": [{"x":0,"y":0,"z":0,"strength_nAm":70}],
  "threshold_snr": 3,
  "modality": "meg" }
```
The current backend ignores the extra fields (harmless). The future backend
should: (a) write `sources` into the config as explicit dipole positions for the
forward solve (instead of geometry-derived `forward.source_spacing_mm` sampling),
(b) build the EEG leadfield too when `modality` includes `eeg`, and (c) run the
detectability calculation after the forward stage.

### 2. New terminal WebSocket message — `result`
Just before `{type:"done"}`, emit:
```json
{ "type": "result",
  "detect": {
    "per_source": [
      {"index":0,"x":0,"y":0,"z":0,"strength_nAm":70,"snr":4.2,"trials_needed":1}
    ],
    "array": {"n_sensors":8190,"mean_snr":4.2,"max_snr":9.1,
              "noise_floor_fT":221.0,"threshold_snr":3.0}
  } }
```
`trials_needed = ceil((threshold_snr / single_trial_snr)^2)`; use `-1` for "never
detectable" (signal ≤ 0). This maps onto `inob.analysis.snr` +
`inob.viz.detectability` (per-source SNR, √n averaging).

### 3. (Optional) `POST /api/detect`
A request/response alternative to the WS `result`, same `detect` shape, body
`{sources, threshold_snr, noise_floor_fT, modality}`. **Until the backend serves
this, the frontend uses a local mock at `app/api/detect/route.ts`** so the UI is
fully demoable; the mock tags its output with `"mocked": true`. Delete that route
once the backend serves `/api/detect` for real (the Next rewrite will then proxy
it through).

## Notes
- The backend CORS allowlist must include the web origin (`http://localhost:3000`)
  if any request is ever made cross-origin; HTTP goes through the Next proxy
  (same-origin) so only the **WebSocket** origin matters.
- Detection threshold and sources are kept client-side (the config schema is
  strict and rejects unknown keys); they reach the backend via the run payload,
  not via `PUT /api/config`.
