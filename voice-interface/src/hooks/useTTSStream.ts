import { useState, useCallback, useEffect, useRef } from 'react';

interface TTSStreamOptions {
  language_code?: string;
  languageCode?: string; // Alias for language_code
  voice?: string;
  sample_rate?: number;
  sampleRate?: number; // Alias for sample_rate
  onAudioChunk?: (chunk: ArrayBuffer) => void;
  onComplete?: () => void;
  onEnd?: () => void; // Alias for onComplete
  onStart?: () => void; // Called when TTS starts
  onError?: (error: Error) => void;
}

type TTSState = {
  isPlaying: boolean;
  isLoading: boolean;
  error: Error | null;
};

export function useTTSStream() {
  const [state, setState] = useState<TTSState>({
    isPlaying: false,
    isLoading: false,
    error: null,
  });

  const audioContextRef = useRef<AudioContext | null>(null);
  const audioQueueRef = useRef<ArrayBuffer[]>([]);
  const isProcessingRef = useRef(false);
  const wsRef = useRef<WebSocket | null>(null);
  const audioBufferSourceRef = useRef<AudioBufferSourceNode | null>(null);

  // Process the audio queue
  const processQueue = useCallback(async () => {
    if (isProcessingRef.current || audioQueueRef.current.length === 0) {
      return;
    }

    isProcessingRef.current = true;
    const audioData = audioQueueRef.current.shift();

    if (!audioData) {
      isProcessingRef.current = false;
      return;
    }

    try {
      const audioContext = audioContextRef.current || new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioContext;

      // Resume audio context if it's suspended
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      const audioBuffer = await audioContext.decodeAudioData(audioData.slice(0));
      
      // Stop any currently playing audio
      if (audioBufferSourceRef.current) {
        audioBufferSourceRef.current.stop();
        audioBufferSourceRef.current.disconnect();
      }

      const source = audioContext.createBufferSource();
      audioBufferSourceRef.current = source;
      
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);

      source.onended = () => {
        if (audioQueueRef.current.length > 0) {
          // Process next chunk in the queue
          processQueue();
        } else {
          // No more chunks to process
          setState((prev) => ({ ...prev, isPlaying: false }));
        }
      };

      source.start();
      setState((prev) => ({ ...prev, isPlaying: true }));
    } catch (error) {
      console.error('Error playing audio:', error);
      setState((prev) => ({
        ...prev,
        error: error instanceof Error ? error : new Error('Failed to play audio'),
        isPlaying: false,
      }));
    } finally {
      isProcessingRef.current = false;
    }
  }, []);

  // Clean up audio resources
  const cleanupAudio = useCallback(() => {
    if (audioBufferSourceRef.current) {
      try {
        audioBufferSourceRef.current.stop();
        audioBufferSourceRef.current.disconnect();
      } catch (e) {
        // Ignore errors when stopping already stopped source
      }
      audioBufferSourceRef.current = null;
    }
    
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
  }, []);

  // Speak function to convert text to speech
  const speak = useCallback(
    async (text: string, options: TTSStreamOptions = {}) => {
      try {
        // Reset state and clean up any existing audio
        cleanupAudio();
        setState({ isPlaying: true, isLoading: true, error: null });
        options.onStart?.();

        // Close any existing WebSocket connection
        if (wsRef.current) {
          wsRef.current.close();
        }

        // Create a new WebSocket connection
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/api/tts/stream`;
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('WebSocket connection established');
          // Send the TTS request
          // Use the new property names with fallback to old ones for backward compatibility
          const languageCode = options.languageCode || options.language_code || 'en-IN';
          const sampleRate = options.sampleRate || options.sample_rate || 24000;
          
          ws.send(
            JSON.stringify({
              text,
              language_code: languageCode,
              voice: options.voice || 'meera',
              sample_rate: sampleRate,
            })
          );
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'audio' && data.data) {
              // Convert base64 to ArrayBuffer
              const binaryString = atob(data.data.audio);
              const len = binaryString.length;
              const bytes = new Uint8Array(len);
              
              for (let i = 0; i < len; i++) {
                bytes[i] = binaryString.charCodeAt(i);
              }
              
              const audioBuffer = bytes.buffer;
              
              // Add to queue and process
              audioQueueRef.current.push(audioBuffer);
              
              // Notify about the new audio chunk if callback is provided
              if (options.onAudioChunk) {
                options.onAudioChunk(audioBuffer);
              }
              
              // Process the queue
              processQueue();
              
            } else if (data.type === 'end') {
              // TTS streaming completed
              ws.close();
              // Call both onComplete and onEnd for backward compatibility
              options.onComplete?.();
              options.onEnd?.();
              
            } else if (data.error) {
              throw new Error(data.error);
            }
            
          } catch (error) {
            console.error('Error processing WebSocket message:', error);
            setState((prev) => ({
              ...prev,
              error: error instanceof Error ? error : new Error('Failed to process TTS response'),
              isPlaying: false,
              isLoading: false,
            }));
            options.onError?.(error instanceof Error ? error : new Error('Failed to process TTS response'));
            ws.close();
          }
        };

        ws.onclose = () => {
          console.log('WebSocket connection closed');
          setState((prev) => ({
            ...prev,
            isPlaying: false,
            isLoading: false,
          }));
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          setState((prev) => ({
            ...prev,
            error: new Error('WebSocket connection error'),
            isPlaying: false,
            isLoading: false,
          }));
          options.onError?.(new Error('WebSocket connection error'));
        };
        
      } catch (error) {
        console.error('Error in speak function:', error);
        setState({
          isPlaying: false,
          isLoading: false,
          error: error instanceof Error ? error : new Error('Failed to initialize TTS'),
        });
        options.onError?.(error instanceof Error ? error : new Error('Failed to initialize TTS'));
      }
    },
    [processQueue, cleanupAudio]
  );

  // Stop TTS playback
  const stop = useCallback(() => {
    // Close WebSocket connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    // Clear audio queue
    audioQueueRef.current = [];
    
    // Stop any currently playing audio
    cleanupAudio();
    
    // Update state
    setState((prev) => ({
      ...prev,
      isPlaying: false,
      isLoading: false,
    }));
  }, [cleanupAudio]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stop();
      cleanupAudio();
    };
  }, [stop, cleanupAudio]);

  return {
    speak,
    stop,
    ...state,
  };
};
