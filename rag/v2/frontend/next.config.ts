import type { NextConfig } from 'next'

const config: NextConfig = {
  output: 'standalone',   // produces self-contained server.js for Docker image

  async rewrites() {
    // In production (Docker), Nginx proxies /api/v2/* to the API container —
    // no rewrite needed. In local dev (npm run dev, API on :8000), Next.js
    // proxies server-side so the browser never makes a cross-origin request.
    if (process.env.NODE_ENV === 'production') {
      return []
    }
    const apiBase = process.env.API_BASE_URL ?? 'http://localhost:8000'
    return [
      {
        source: '/api/v2/:path*',
        destination: `${apiBase}/api/v2/:path*`,
      },
    ]
  },
}

export default config
