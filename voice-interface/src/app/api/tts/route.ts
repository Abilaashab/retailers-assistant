import { NextResponse } from 'next/server';
import { SARVAM_CONFIG, toBcp47Code } from '@/config';

// Function to clean text for TTS by removing emojis and certain special characters
export function cleanTextForTTS(text: string): string {
  // Remove emojis and other special characters but keep basic punctuation
  return text
    // Remove emojis
    .replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, '')
    // Remove markdown formatting
    .replace(/\*([^*]+)\*/g, '$1')
    // Replace URLs with "Read more"
    .replace(/Read more: https?:\/\/[^\s]+/g, 'Read more')
    // Remove other special characters that might affect TTS
    .replace(/[\u2018\u2019]/g, "'") // Replace smart quotes with straight quotes
    .replace(/[\u201C\u201D]/g, '"')
    // Remove colons and semicolons that aren't part of time or other meaningful patterns
    .replace(/(?<!(\d{1,2}:\d{2})|(https?:\/\/)|(mailto:))[:;]/g, '')
    // Remove multiple spaces and trim
    .replace(/\s+/g, ' ')
    .trim();
}

// Get API key from environment variables
const apiKey = process.env.SARVAM_AI_API || process.env.NEXT_PUBLIC_SARVAM_AI_API;

type TTSRequest = {
  text: string;
  language_code?: string;
  voice?: string;  // Changed from voice_id to voice
  sample_rate?: number;
};

export async function POST(request: Request) {
  try {
    if (!apiKey) {
      const errorMsg = 'Sarvam AI API key is not set';
      console.error(errorMsg);
      return NextResponse.json(
        { error: 'Server configuration error: API key not set' },
        { status: 500 }
      );
    }

    const requestData = await request.json();
    let { text, language_code, voice, sample_rate }: TTSRequest = requestData;  // Changed from voice_id to voice

    if (!text) {
      const errorMsg = 'Text is required for text-to-speech conversion';
      console.error(errorMsg);
      return NextResponse.json(
        { error: errorMsg },
        { status: 400 }
      );
    }

    // Clean the text for TTS
    text = cleanTextForTTS(text);

    // Get the base language code (e.g., 'en' from 'en-IN')
    const baseLang = language_code ? language_code.split('-')[0].toLowerCase() : 'en';
    
    // Map of language codes to their supported voices
    const allVoices = [
      'diya', 'maya', 'meera', 'pavithra', 'maitreyi', 'misha',  // Female voices
      'amol', 'arjun', 'amartya', 'arvind', 'neel', 'vian'      // Male voices
    ];
    
    // All voices are available for all languages
    const languageToVoices: Record<string, string[]> = {
      'en': allVoices,
      'hi': allVoices,
      'ta': allVoices,
      'te': allVoices,
      'kn': allVoices,
      'ml': allVoices,
      'bn': allVoices,
      'gu': allVoices,
      'mr': allVoices,
      'pa': allVoices
    };
    
    // Get default voice for the language if not provided or invalid
    let voiceToUse = voice?.toLowerCase() || 'meera';
    const validVoices = languageToVoices[baseLang] || ['meera'];
    
    if (!validVoices.includes(voiceToUse)) {
      console.warn(`Voice ${voiceToUse} is not available for language ${baseLang}, using default`);
      voiceToUse = validVoices[0];
    }

    // Format language code if needed (e.g., 'en' -> 'en-IN')
    const formatLanguageCode = (lang: string): string => {
      if (!lang) return 'en-IN';
      if (lang.includes('-')) return lang;
      return `${lang}-IN`;
    };

    const ttsUrl = `${SARVAM_CONFIG.BASE_URL}${SARVAM_CONFIG.ENDPOINTS.TEXT_TO_SPEECH}`;
    
    // Format the language code to BCP-47 format (e.g., 'en' -> 'en-IN')
    const targetLanguageCode = formatLanguageCode(language_code || 'en');
    
    // Format the text by adding commas to numbers for better pronunciation
    const formattedText = text.replace(/(\d{5,})/g, (match) => {
      return match.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    });

    const requestBody = {
      text: formattedText,
      target_language_code: targetLanguageCode,
      speaker: voiceToUse,
      model: SARVAM_CONFIG.DEFAULTS.TTS_MODEL,
      speech_sample_rate: sample_rate || SARVAM_CONFIG.DEFAULTS.SAMPLE_RATE,
      pace: 1.0, // Normal speed
      loudness: 1.0, // Slightly louder for better clarity
      pitch: 0, // Normal pitch
      enable_preprocessing: true,
    };

    try {
      const response = await fetch(ttsUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          [SARVAM_CONFIG.API_KEY_HEADER]: apiKey,
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Failed to read error response');
        console.error('TTS API Error Response:', {
          status: response.status,
          statusText: response.statusText,
          error: errorText,
          headers: Object.fromEntries(response.headers.entries()),
        });
        throw new Error(`TTS API request failed with status ${response.status}: ${response.statusText}`);
      }

      try {
        // Parse the JSON response
        const jsonResponse = await response.json();
        
        // Handle the Sarvam API response format
        if (!jsonResponse.audios || !Array.isArray(jsonResponse.audios) || jsonResponse.audios.length === 0) {
          throw new Error('No audio data found in response');
        }
        
        // Get the first audio from the audios array
        const audioData = jsonResponse.audios[0];
        
        if (!audioData) {
          throw new Error('Empty audio data in response');
        }
        
        // Return the audio data in the expected format
        return new NextResponse(JSON.stringify({ 
          audio: audioData,
          requestId: jsonResponse.request_id
        }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      } catch (processError) {
        console.error('Error processing TTS response:', processError);
        throw new Error(`Failed to process TTS response: ${processError instanceof Error ? processError.message : 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error in TTS API call:', error);
      return NextResponse.json(
        { 
          error: 'Failed to process text-to-speech request',
          details: error instanceof Error ? error.message : String(error)
        },
        { status: 500 }
      );
    }
    
  } catch (error) {
    console.error('Error in TTS API route:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { 
        error: 'Internal server error',
        details: errorMessage,
      },
      { status: 500 }
    );
  }
}
