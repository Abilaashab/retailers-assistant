import { NextResponse } from 'next/server';
import { SARVAM_CONFIG, toBcp47Code } from '@/config';

// Get API key from environment variables
const apiKey = process.env.SARVAM_AI_API || process.env.NEXT_PUBLIC_SARVAM_AI_API;

type ChatRequest = {
  text?: string;
  sourceLang?: string;
  targetLang?: string;
  isAudio?: boolean;
  audioData?: string; // Base64 encoded audio data
  audioFormat?: string; // e.g., 'audio/wav', 'audio/mp3'
  skipTranslation?: boolean; // Flag to skip translation for English inputs
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
  original_text?: string;
  agent_response?: string;
};

type SupervisorResponse = {
  response: string;
  error?: string;
};

async function processText(text: string, sourceLang: string, targetLang: string): Promise<ChatResponse> {
  if (!apiKey) {
    throw new Error('API key not configured');
  }
  
  // Debug: Log API key status (redacted for security)
  console.log('API Key configured:', apiKey ? '[REDACTED]' : 'Not found');
  console.log('API Key header name:', SARVAM_CONFIG.API_KEY_HEADER);

  // Convert language codes to the format expected by Sarvam AI
  const formatLanguageCode = (lang: string): string => {
    // If already in xx-XX format, return as is
    if (lang.includes('-')) return lang;
    
    // Map common language codes to their full format
    const langMap: Record<string, string> = {
      'en': 'en-IN',
      'hi': 'hi-IN',
      'ta': 'ta-IN',
      'te': 'te-IN',
      'kn': 'kn-IN',
      'ml': 'ml-IN',
      'bn': 'bn-IN',
      'gu': 'gu-IN',
      'mr': 'mr-IN',
      'pa': 'pa-IN',
      'or': 'or-IN',
      'as': 'as-IN'
    };
    
    return langMap[lang] || 'en-IN'; // Default to English if language not found
  };
  
  const sourceLangCode = formatLanguageCode(sourceLang);
  const targetLangCode = formatLanguageCode(targetLang);

  // If source and target languages are the same, return the text as is
  if (sourceLangCode === targetLangCode) {
    return {
      transcript: text,
      translated_text: text,
      source_lang: sourceLangCode,
      target_lang: targetLangCode,
      is_simulated: true
    };
  }

  // Use the direct API endpoint for translation
  const apiUrl = 'https://api.sarvam.ai/translate';
  
  // Prepare the request body according to the API spec
  const requestBody = {
    input: text,
    source_language_code: sourceLangCode,
    target_language_code: targetLangCode,
    // Optional parameters with defaults
    speaker_gender: 'Male',
    mode: 'modern-colloquial',
    enable_preprocessing: true,
    numerals_format: 'international',
    // Add the API key to the headers instead of the body
  };

  const headers = {
    'Content-Type': 'application/json',
    [SARVAM_CONFIG.API_KEY_HEADER]: apiKey,
    'Accept': 'application/json'
  };

  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: headers,
    body: JSON.stringify(requestBody)
  });

  const responseData = await response.json();
  
  if (!response.ok) {
    console.error('Sarvam API error status:', response.status);
    console.error('Sarvam API error response:', responseData);
    throw new Error(`API request failed with status ${response.status}: ${responseData?.message || 'Unknown error'}`);
  }

  return {
    transcript: text,
    translated_text: responseData.translated_text || `[No translation returned]`,
    source_lang: responseData.source_language_code || sourceLangCode,
    target_lang: targetLangCode,
    is_simulated: false,
    request_id: responseData.request_id,
    language_code: responseData.source_language_code
  };
}

// Function to call the supervisor agent
async function callSupervisorAgent(query: string): Promise<SupervisorResponse> {
  try {
    // Assuming the supervisor agent is running on localhost:8000
    const response = await fetch('http://localhost:8000/process_query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('Supervisor agent error:', error);
      return { 
        response: "I'm sorry, I'm having trouble connecting to the assistant service.",
        error: error
      };
    }

    return await response.json();
  } catch (error) {
    console.error('Error calling supervisor agent:', error);
    return {
      response: "I'm sorry, I encountered an error processing your request.",
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

async function processAudio(audioData: string, audioFormat: string, sourceLang: string, targetLang: string): Promise<ChatResponse> {
  if (!apiKey) {
    throw new Error('API key not configured');
  }
  
  // Debug: Log API key status (redacted for security)
  console.log('API Key configured:', apiKey ? '[REDACTED]' : 'Not found');
  console.log('API Key header name:', SARVAM_CONFIG.API_KEY_HEADER);

  // Use the direct API endpoint for speech-to-text translation
  const apiUrl = 'https://api.sarvam.ai/speech/translate';
  // Format language codes for Sarvam AI
  const formatLanguageCode = (lang: string): string => {
    if (lang.includes('-')) return lang;
    
    const langMap: Record<string, string> = {
      'en': 'en-IN',
      'hi': 'hi-IN',
      'ta': 'ta-IN',
      'te': 'te-IN',
      'kn': 'kn-IN',
      'ml': 'ml-IN',
      'bn': 'bn-IN',
      'gu': 'gu-IN',
      'mr': 'mr-IN',
      'pa': 'pa-IN',
      'or': 'or-IN',
      'as': 'as-IN'
    };
    
    return langMap[lang] || 'en-IN';
  };
  
  const sourceLangCode = formatLanguageCode(sourceLang);
  const targetLangCode = formatLanguageCode(targetLang);
  

  
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
  formData.append('source_language', sourceLangCode);
  formData.append('target_language', targetLangCode);
  formData.append('speaker_gender', 'Male');
  formData.append('mode', 'formal');
  formData.append('enable_preprocessing', 'true');
  formData.append('numerals_format', 'international');

  const headers = {
    [SARVAM_CONFIG.API_KEY_HEADER]: apiKey,
  };

  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: headers,
    body: formData,
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
    source_lang: responseData.language_code || sourceLangCode,
    target_lang: targetLangCode,
    is_simulated: false,
    request_id: responseData.request_id,
    language_code: responseData.language_code,
    original_text: responseData.transcript // Include original transcript
  };
}

export async function POST(request: Request) {
  try {
    // Parse the request body
    const requestData: ChatRequest = await request.json();
    const { text, sourceLang, targetLang = 'en', isAudio = false, audioData, audioFormat, skipTranslation = false } = requestData;



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

        result = await processAudio(audioData, audioFormat || 'audio/wav', sourceLang || 'en', 'en');
      } else if (text) {

        
        const effectiveSourceLang = sourceLang || 'en';
        // Skip translation if source language is English or explicitly requested
        if (skipTranslation || effectiveSourceLang.startsWith('en')) {

          result = {
            transcript: text,
            translated_text: text, // Use original text as translated text
            source_lang: 'en-IN',
            target_lang: 'en-IN',
            is_simulated: false,
            request_id: 'skipped-translation',
            language_code: 'en-IN'
          };
        } else {
          result = await processText(text, effectiveSourceLang, 'en');
        }
      } else {
        throw new Error('No text or audio data provided');
      }
      

      
      // Call supervisor agent with the translated text
      let agentResponse;
      try {
        agentResponse = await callSupervisorAgent(result.translated_text);

        
        if (agentResponse.error) {
          throw new Error(agentResponse.error);
        }
        
        // If the target language is not English, try to translate the agent's response back
        let finalResponse = agentResponse.response;
        if (targetLang && targetLang !== 'en') {
          try {

            const translatedResponse = await processText(agentResponse.response, 'en', targetLang);
            if (translatedResponse && translatedResponse.translated_text) {
              finalResponse = translatedResponse.translated_text;

            }
          } catch (translationError) {
            console.error('Error translating response:', translationError);
            // Continue with English response if translation fails
            finalResponse = agentResponse.response;
          }
        }
        
        result = {
          ...result,
          original_text: text,
          agent_response: finalResponse,
          target_lang: targetLang
        };
        
        return NextResponse.json(result);
        
      } catch (agentError) {
        console.error('Error calling supervisor agent:', agentError);
        // Return the original translation result with an error message
        return NextResponse.json({
          ...result,
          error: agentError instanceof Error ? agentError.message : 'Error processing request',
          agent_response: agentError instanceof Error ? agentError.message : 'Error processing request'
        });
      }
      
    } catch (apiError) {
      console.error('Error processing request:', apiError);
      
      // Fallback to simulated response if API call fails
      // Fallback to simulated response
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
