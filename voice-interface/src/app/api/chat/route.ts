import { NextResponse } from 'next/server';
import { SARVAM_CONFIG, toBcp47Code } from '@/config';

// Get API key from environment variables
const apiKey = process.env.SARVAM_AI_API || process.env.NEXT_PUBLIC_SARVAM_AI_API;

if (!apiKey) {
  console.error('Sarvam AI API key is not set. Please set SARVAM_AI_API environment variable.');
}

type ChatRequest = {
  text: string;
  sourceLang: string;
  targetLang: string;
  isAudio?: boolean;
  audioData?: string; // Base64 encoded audio data
  audioFormat?: string; // e.g., 'audio/wav', 'audio/mp3'
};

type SarvamSpeechToTextResponse = {
  transcript: string;
  request_id: string;
  language_code: string;
  diarized_transcript?: {
    entries: Array<{
      transcript: string;
      start_time_seconds: number;
      end_time_seconds: number;
      speaker_id: string;
    }>;
  };
};

type ChatResponse = {
  transcript: string;
  translated_text: string;
  source_lang: string;
  target_lang: string;
  is_simulated: boolean;
  request_id?: string;
  language_code?: string;
  error?: string;
};

async function processText(text: string, sourceLang: string, targetLang: string): Promise<ChatResponse> {
  if (!apiKey) {
    throw new Error('API key not configured');
  }

  const apiUrl = `${SARVAM_CONFIG.BASE_URL}${SARVAM_CONFIG.ENDPOINTS.TEXT_TO_TEXT_TRANSLATE}`;
  const bcp47SourceLang = toBcp47Code(sourceLang);
  const bcp47TargetLang = toBcp47Code(targetLang);

  console.log(`Translating text from ${bcp47SourceLang} to ${bcp47TargetLang}`);
  
  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      [SARVAM_CONFIG.API_KEY_HEADER]: apiKey,
      'Accept': 'application/json'
    },
    body: JSON.stringify({
      text: text,
      source_language: bcp47SourceLang,
      target_language: bcp47TargetLang
    })
  });

  const responseData = await response.json();
  
  if (!response.ok) {
    console.error('Sarvam API error status:', response.status);
    console.error('Sarvam API error response:', responseData);
    throw new Error(`API request failed with status ${response.status}: ${responseData?.message || 'Unknown error'}`);
  }

  return {
    transcript: text,
    translated_text: responseData.translated_text || responseData.text || `[No translation returned]`,
    source_lang: bcp47SourceLang,
    target_lang: bcp47TargetLang,
    is_simulated: false,
    request_id: responseData.request_id,
    language_code: responseData.language_code
  };
}

async function processAudio(audioData: string, audioFormat: string, sourceLang: string, targetLang: string): Promise<ChatResponse> {
  if (!apiKey) {
    throw new Error('API key not configured');
  }

  const apiUrl = `${SARVAM_CONFIG.BASE_URL}${SARVAM_CONFIG.ENDPOINTS.SPEECH_TO_TEXT_TRANSLATE}`;
  const bcp47SourceLang = toBcp47Code(sourceLang);
  const bcp47TargetLang = toBcp47Code(targetLang);
  
  console.log(`Processing audio from ${bcp47SourceLang} to ${bcp47TargetLang}`);
  
  // Convert base64 to blob
  const byteString = atob(audioData.split(',')[1]);
  const mimeString = audioFormat || 'audio/wav';
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  
  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }
  
  const blob = new Blob([ab], { type: mimeString });
  
  const formData = new FormData();
  formData.append('file', blob, 'recording.wav');
  formData.append('source_language', bcp47SourceLang);
  formData.append('target_language', bcp47TargetLang);

  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      [SARVAM_CONFIG.API_KEY_HEADER]: apiKey,
    },
    body: formData
  });

  const responseData: SarvamSpeechToTextResponse = await response.json();
  
  if (!response.ok) {
    console.error('Sarvam API error status:', response.status);
    console.error('Sarvam API error response:', responseData);
    throw new Error(`API request failed with status ${response.status}`);
  }

  // Use the transcript directly as the translated text
  // The translation should be handled by the API if source and target languages are different
  return {
    transcript: responseData.transcript,
    translated_text: responseData.transcript,
    source_lang: responseData.language_code || 'en-IN',
    target_lang: bcp47TargetLang,
    is_simulated: false,
    request_id: responseData.request_id,
    language_code: responseData.language_code
  };
}

export async function POST(request: Request) {
  try {
    // Parse the request body
    const requestData: ChatRequest = await request.json();
    const { text, sourceLang, targetLang = 'en', isAudio = false, audioData, audioFormat } = requestData;

    console.log('Received chat request');
    console.log('Request type:', isAudio ? 'Audio' : 'Text');
    console.log('Source Lang:', sourceLang);
    console.log('Target Lang:', targetLang);

    if (!apiKey) {
      console.error('Sarvam AI API key is not set');
      return NextResponse.json(
        { error: 'Server configuration error: API key not set' },
        { status: 500 }
      );
    }

    if (isAudio && !audioData) {
      return NextResponse.json(
        { error: 'Audio data is required for audio requests' },
        { status: 400 }
      );
    }

    if (!isAudio && !text) {
      return NextResponse.json(
        { error: 'Text is required for text requests' },
        { status: 400 }
      );
    }

    try {
      let result: ChatResponse;
      
      if (isAudio && audioData) {
        result = await processAudio(audioData, audioFormat || 'audio/wav', sourceLang || 'en', targetLang);
      } else if (text) {
        result = await processText(text, sourceLang || 'en', targetLang);
      } else {
        throw new Error('Invalid request: either text or audio data must be provided');
      }
      
      return NextResponse.json(result);
      
    } catch (apiError) {
      console.error('Error processing request:', apiError);
      
      // Fallback to simulated response if API call fails
      console.warn('Falling back to simulated response');
      const errorMessage = apiError instanceof Error ? apiError.message : 'Unknown error';
      const simulatedResponse: ChatResponse = {
        transcript: isAudio ? '[Audio processing failed]' : text || '',
        translated_text: isAudio 
          ? `[Simulated Audio Translation to ${targetLang}]: Audio processing failed - ${errorMessage}`
          : `[Simulated Translation from ${sourceLang} to ${targetLang}]: ${text || ''}`,
        source_lang: sourceLang || 'en',
        target_lang: targetLang,
        is_simulated: true,
        error: errorMessage
      };
      
      return NextResponse.json(simulatedResponse);
    }
    
  } catch (error) {
    console.error('Error processing request:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { 
        error: 'Internal server error', 
        details: errorMessage,
        is_simulated: true
      },
      { status: 500 }
    );
  }
}
