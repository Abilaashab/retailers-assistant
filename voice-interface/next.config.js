/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    // This makes the environment variables available on the client side
    // Only include variables that are safe to expose to the browser
    NEXT_PUBLIC_SARVAM_AI_API: process.env.SARVAM_AI_API,
  },
  // This makes environment variables available to the server-side code
  serverRuntimeConfig: {
    // Will only be available on the server side
    SARVAM_AI_API: process.env.SARVAM_AI_API,
  },
  // This makes environment variables available to the client-side code
  publicRuntimeConfig: {
    // Will be available on both server and client
    NEXT_PUBLIC_SARVAM_AI_API: process.env.SARVAM_AI_API,
  },
};

module.exports = nextConfig;
