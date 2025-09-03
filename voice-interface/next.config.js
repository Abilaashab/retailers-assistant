/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  
  // Environment variables configuration
  env: {
    // Make sure the API key is available on both client and server
    NEXT_PUBLIC_SARVAM_AI_API: process.env.SARVAM_AI_API,
  },
  
  // Webpack configuration
  webpack: (config, { isServer }) => {
    // Add fallbacks for Node.js modules that might be required
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        net: false,
        tls: false,
        fs: false,
        dns: false,
        child_process: false,
      };
    }
    
    // Use isomorphic-ws for WebSocket support
    config.resolve.alias = {
      ...config.resolve.alias,
      'ws': 'isomorphic-ws',
    };
    
    return config;
  },
  
  // CORS headers for API routes
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Credentials', value: 'true' },
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET,OPTIONS,PATCH,DELETE,POST,PUT' },
          { key: 'Access-Control-Allow-Headers', value: 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version' },
        ],
      },
    ];
  },
  
  // Development server configuration
  webpackDevMiddleware: (config) => {
    // Configure watch options for better development experience
    config.watchOptions = {
      ...config.watchOptions,
      poll: 1000,
      ignored: ['node_modules/**', '.next/**'],
    };
    return config;
  },
  
  // Enable experimental features
  experimental: {
    serverComponentsExternalPackages: ['ws'],
  },
};

// Only run WebSocket server in development
if (process.env.NODE_ENV === 'development') {
  const { parse } = require('url');
  const { WebSocketServer } = require('ws');
  const { createServer } = require('http');
  
  // Create an HTTP server for WebSocket connections
  const server = createServer();
  const wss = new WebSocketServer({ noServer: true });
  
  // Handle WebSocket connections
  wss.on('connection', (ws) => {
    console.log('WebSocket connection established');
    
    ws.on('message', (message) => {
      console.log('Received message:', message.toString());
      ws.send(`Echo: ${message}`);
    });
    
    ws.on('close', () => {
      console.log('WebSocket connection closed');
    });
  });
  
  // Handle HTTP server upgrade requests
  server.on('upgrade', (request, socket, head) => {
    const { pathname } = parse(request.url);
    
    if (pathname === '/api/tts/stream') {
      wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit('connection', ws, request);
      });
    } else {
      socket.destroy();
    }
  });
  
  // Start the server on a different port than Next.js
  const PORT = 3001;
  server.listen(PORT, () => {
    console.log(`WebSocket server is running on ws://localhost:${PORT}`);
  });
}

module.exports = nextConfig;
