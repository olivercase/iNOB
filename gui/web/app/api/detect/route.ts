// DEV MOCK fallback for the detectability calculation.
//
// This exists so the UI is fully demoable before the Python backend implements
// the real endpoint (see API_CONTRACT.md). It is NOT the real physics: it uses
// a crude leadfield surrogate (signal falls off with depth from a nominal
// sensor shell) purely to produce plausible-looking numbers. When the backend
// emits a {type:"result"} WebSocket message, this route is unused.
//
// As soon as the backend serves /api/detect for real, delete this file — the
// Next rewrite will then proxy /api/detect to FastAPI.

import { NextResponse } from "next/server";

interface Body {
  sources: { x: number; y: number; z: number; strength_nAm: number }[];
  threshold_snr: number;
  noise_floor_fT: number;
}

// Crude surrogate: fT at best channel ~ k * Q / r^2, r = distance to a nominal
// sensor shell ~60 mm from the neck axis. Single-trial SNR = signal / noise.
const K_MOCK = 1.2e5; // fT * mm^2 per nA·m — order-of-magnitude only

export async function POST(req: Request) {
  const body = (await req.json()) as Body;
  const noise = Math.max(body.noise_floor_fT || 1, 1e-6);
  const thr = Math.max(body.threshold_snr || 3, 1e-6);

  const per_source = (body.sources ?? []).map((s, index) => {
    const r = Math.max(Math.hypot(s.x, s.y) - 60, 8); // mm to sensor shell
    const signal_fT = (K_MOCK * (s.strength_nAm || 1)) / (r * r);
    const snr = signal_fT / noise;
    const trials_needed = snr > 0 ? Math.ceil((thr / snr) ** 2) : Infinity;
    return {
      index,
      x: s.x,
      y: s.y,
      z: s.z,
      strength_nAm: s.strength_nAm,
      snr: Number(snr.toFixed(3)),
      trials_needed: Number.isFinite(trials_needed) ? trials_needed : -1,
    };
  });

  const snrs = per_source.map((p) => p.snr);
  return NextResponse.json({
    per_source,
    array: {
      n_sensors: 8190,
      mean_snr: snrs.length ? Number((snrs.reduce((a, b) => a + b, 0) / snrs.length).toFixed(3)) : 0,
      max_snr: snrs.length ? Math.max(...snrs) : 0,
      noise_floor_fT: noise,
      threshold_snr: thr,
    },
    mocked: true,
  });
}
