// API Configuration for Sarvam AI
export const SARVAM_CONFIG = {
  // Base URL for all API endpoints
  BASE_URL: 'https://api.sarvam.ai',
  
  // API Key header name (as per Sarvam AI documentation)
  API_KEY_HEADER: 'api-subscription-key',
  
  // API Endpoints
  ENDPOINTS: {
    // Text to Text Translation
    TEXT_TO_TEXT_TRANSLATE: '/v1/translate',
    
    // Speech to Text with Translation
    SPEECH_TO_TEXT_TRANSLATE: '/v1/transcribe/translate',
    
    // Text to Speech
    TEXT_TO_SPEECH: '/v1/tts',
  },
  
  // Supported languages with their display names and language codes
  LANGUAGES: [
    { code: 'en', name: 'English' },
    { code: 'hi', name: 'हिंदी (Hindi)' },
    { code: 'mr', name: 'मराठी (Marathi)' },
    { code: 'ta', name: 'தமிழ் (Tamil)' },
    { code: 'te', name: 'తెలుగు (Telugu)' },
    { code: 'kn', name: 'ಕನ್ನಡ (Kannada)' },
    { code: 'ml', name: 'മലയാളം (Malayalam)' },
    { code: 'bn', name: 'বাংলা (Bengali)' },
    { code: 'gu', name: 'ગુજરાતી (Gujarati)' },
    { code: 'pa', name: 'ਪੰਜਾਬੀ (Punjabi)' },
    { code: 'or', name: 'ଓଡ଼ିଆ (Odia)' },
    { code: 'as', name: 'অসমীয়া (Assamese)' },
  ],
  
  // Default settings
  DEFAULTS: {
    TARGET_LANGUAGE: 'en',
    SOURCE_LANGUAGE: 'hi',
    SPEAKER_GENDER: 'female',
    VOICE_ID: 'default',
    SAMPLE_RATE: 24000,
  },
};

// Helper function to convert language code to BCP-47 format
export function toBcp47Code(langCode: string): string {
  // Add any necessary language code mappings here
  const codeMappings: Record<string, string> = {
    // Add any non-standard language code mappings here
    // Example: 'hindi': 'hi-IN',
  };
  
  return codeMappings[langCode.toLowerCase()] || langCode;
}

// Helper function to get language name from code
export function getLanguageName(code: string): string {
  const lang = SARVAM_CONFIG.LANGUAGES.find(lang => 
    lang.code.toLowerCase() === code.toLowerCase()
  );
  return lang ? lang.name : code;
}
