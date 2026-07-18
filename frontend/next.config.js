/** @type {import('next').NextConfig} */
const nextConfig = {
  // API proxying lives in a Node route handler. Vercel service bindings exist
  // at function runtime, so they must not be read in build-time rewrites.
};

module.exports = nextConfig;
