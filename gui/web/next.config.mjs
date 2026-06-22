/** @type {import('next').NextConfig} */
const BACKEND = process.env.INOB_BACKEND_HTTP || "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
  // HTTP /api/* is proxied to the FastAPI backend server-side (avoids CORS).
  // Local route handlers (e.g. app/api/detect) take precedence over these
  // afterFiles rewrites, so the dev mock fallback still works.
  async rewrites() {
    return [
      // health/config/meshes -> backend. /api/run is a WebSocket and is
      // connected to directly from the client (see lib/api.ts).
      { source: "/api/health", destination: `${BACKEND}/api/health` },
      { source: "/api/config/:path*", destination: `${BACKEND}/api/config/:path*` },
      { source: "/api/config", destination: `${BACKEND}/api/config` },
      { source: "/api/meshes/:path*", destination: `${BACKEND}/api/meshes/:path*` },
      { source: "/api/meshes", destination: `${BACKEND}/api/meshes` },
      { source: "/api/cluster/:path*", destination: `${BACKEND}/api/cluster/:path*` },
    ];
  },
};

export default nextConfig;
