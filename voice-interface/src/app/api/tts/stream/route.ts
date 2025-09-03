import { WebSocketServer, WebSocket } from 'ws';
import { NextRequest, NextResponse } from 'next/server';
import { SARVAM_CONFIG } from '@/config';
import { IncomingMessage } from 'http';
import { Duplex } from 'stream';

// Re-export the cleanTextForTTS function from the parent route
import { cleanTextForTTS } from '../route';

// Extend the global type to include our WebSocket server
declare global {
  var wss: WebSocketServer | undefined;
}

// Initialize WebSocket server if it doesn't exist
const wss = global.wss || new WebSocketServer({ noServer: true });
if (process.env.NODE_ENV !== 'production') {
  global.wss = wss;
}

// Get API key from environment variables
const apiKey = process.env.SARVAM_AI_API || process.env.NEXT_PUBLIC_SARVAM_AI_API;

type TTSStreamRequest = {
  text: string;
  language_code?: string;
  voice?: string;
  sample_rate?: number;
  onAudioChunk?: (chunk: ArrayBuffer) => void;
  onComplete?: () => void;
  onError?: (error: Error) => void;
};

// Track active connections
const activeConnections = new Map<string, WebSocket>();

// Generate a unique ID for each connection
const generateConnectionId = () => {
  return Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
};

// Format language code if needed (e.g., 'en' -> 'en-IN')
const formatLanguageCode = (lang: string): string => {
  if (!lang) return 'en-IN';
  if (lang.includes('-')) return lang;
  return `${lang}-IN`;
};

export async function GET(request: NextRequest) {
  if (!apiKey) {
    return new NextResponse('Sarvam AI API key is not set', { status: 500 });
  }

  // Check if this is a WebSocket upgrade request
  if (request.headers.get('upgrade') === 'websocket') {
    // Handle WebSocket upgrade
    const { socket, response } = await new Promise<{ socket: WebSocket; response: Response }>((resolve) => {
      // Create a new WebSocket server for this connection
      const server = new WebSocketServer({ noServer: true });
      
      server.on('connection', (ws) => {
        const connectionId = generateConnectionId();
        activeConnections.set(connectionId, ws);
        
        // Handle WebSocket messages
        ws.on('message', async (message) => {
          try {
            const data = JSON.parse(message.toString()) as TTSStreamRequest;
            
            if (!data.text) {
              ws.send(JSON.stringify({ error: 'Text is required for text-to-speech conversion' }));
              return;
            }
            
            // Clean the text for TTS
            const cleanedText = cleanTextForTTS(data.text);
            const targetLanguageCode = formatLanguageCode(data.language_code || 'en');
            const voiceToUse = data.voice || 'meera';
            
            // Format the text by adding commas to numbers for better pronunciation
            const formattedText = cleanedText.replace(/(\d{5,})/g, (match) => {
              return match.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
            });
            
            // Create WebSocket connection to Sarvam AI
            const sarvamWs = new WebSocket('wss://api.sarvam.ai/v1/tts/stream');
            
            sarvamWs.on('open', () => {
              console.log('Connected to Sarvam AI TTS streaming API');
              
              // Send configuration
              sarvamWs.send(JSON.stringify({
                type: 'config',
                data: {
                  speaker: voiceToUse,
                  target_language_code: targetLanguageCode,
                  min_buffer_size: 50,
                  max_chunk_length: 200,
                  output_audio_codec: 'mp3',
                  output_audio_bitrate: '128k',
                  pace: 1.0,
                  pitch: 0,
                  loudness: 1.0
                }
              }));
              
              // Send text in chunks if it's too long
              const maxChunkSize = 500;
              for (let i = 0; i < formattedText.length; i += maxChunkSize) {
                const chunk = formattedText.substring(i, i + maxChunkSize);
                sarvamWs.send(JSON.stringify({
                  type: 'text',
                  data: { text: chunk }
                }));
              }
              
              // Send flush to ensure all text is processed
              sarvamWs.send(JSON.stringify({ type: 'flush' }));
            });
            
            sarvamWs.on('message', (wsData: Buffer) => {
              try {
                const message = JSON.parse(wsData.toString());
                
                if (message.type === 'audio' && message.data) {
                  // Forward audio chunks to the client
                  ws.send(JSON.stringify({
                    type: 'audio',
                    data: message.data
                  }));
                  
                  // Call the onAudioChunk callback if provided
                  if (data.onAudioChunk) {
                    const binaryString = atob(message.data.audio);
                    const len = binaryString.length;
                    const bytes = new Uint8Array(len);
                    
                    for (let i = 0; i < len; i++) {
                      bytes[i] = binaryString.charCodeAt(i);
                    }
                    
                    data.onAudioChunk(bytes.buffer);
                  }
                } else if (message.type === 'end') {
                  ws.send(JSON.stringify({ type: 'end' }));
                  data.onComplete?.();
                } else if (message.error) {
                  throw new Error(message.error);
                }
              } catch (err) {
                console.error('Error processing WebSocket message:', err);
                ws.send(JSON.stringify({
                  error: 'Failed to process TTS response',
                  details: err instanceof Error ? err.message : 'Unknown error'
                }));
                ws.close();
              }
            });
            
            sarvamWs.on('close', () => {
              console.log('Disconnected from Sarvam AI TTS streaming API');
              ws.send(JSON.stringify({ type: 'end' }));
              ws.close();
            });
            
            sarvamWs.on('error', (error) => {
              console.error('WebSocket error:', error);
              ws.send(JSON.stringify({ error: 'WebSocket connection error' }));
              ws.close();
            });
            
            // Handle client disconnection
            ws.on('close', () => {
              console.log('Client disconnected');
              activeConnections.delete(connectionId);
              sarvamWs.close();
            });
            
          } catch (error) {
            console.error('Error processing TTS request:', error);
            ws.send(JSON.stringify({
              error: 'Failed to process TTS request',
              details: error instanceof Error ? error.message : 'Unknown error'
            }));
            ws.close();
          }
        });
        
        ws.on('error', (error) => {
          console.error('WebSocket error:', error);
          activeConnections.delete(connectionId);
          ws.close();
        });
      });
      
      // Handle the upgrade
      const req = request as unknown as IncomingMessage;
      const socket = (request as any).socket as unknown as Duplex;
      const head = Buffer.alloc(0);
      
      server.handleUpgrade(req, socket, head, (ws) => {
        server.emit('connection', ws, req);
      });
      
      // Create a response that will be used to complete the upgrade
      resolve({
        socket: null as unknown as WebSocket, // This is just to satisfy the type
        response: new NextResponse(null, { 
          status: 101,
          headers: {
            'Upgrade': 'websocket',
            'Connection': 'Upgrade',
          }
        })
      });
    });
    
    return response;
  }

  // For non-WebSocket requests, return a 400 error
  return new NextResponse('Expected WebSocket upgrade request', { status: 400 });
}
