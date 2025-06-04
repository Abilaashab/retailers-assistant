// Voice configuration
export const VOICES = {
  female: [
    { id: 'diya', name: 'Diya' },
    { id: 'maya', name: 'Maya' },
    { id: 'meera', name: 'Meera' },
    { id: 'pavithra', name: 'Pavithra' },
    { id: 'maitreyi', name: 'Maitreyi' },
    { id: 'misha', name: 'Misha' },
  ],
  male: [
    { id: 'amol', name: 'Amol' },
    { id: 'arjun', name: 'Arjun' },
    { id: 'amartya', name: 'Amartya' },
    { id: 'arvind', name: 'Arvind' },
    { id: 'neel', name: 'Neel' },
    { id: 'vian', name: 'Vian' },
  ],
} as const;

// API Configuration for Sarvam AI
export const SARVAM_CONFIG = {
  // Base URL for all API endpoints
  BASE_URL: 'https://api.sarvam.ai',
  
  // API Key header name (as per Sarvam AI documentation)
  API_KEY_HEADER: 'api-subscription-key',
  
  // API Endpoints
  ENDPOINTS: {
    // Text to Text Translation
    TEXT_TO_TEXT_TRANSLATE: '/translate',
    
    // Speech to Text with Translation
    SPEECH_TO_TEXT_TRANSLATE: '/transcribe/translate',
    
    // Text to Speech
    TEXT_TO_SPEECH: '/text-to-speech',
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
    SPEAKER: 'meera', // Default speaker voice (must be lowercase)
    TTS_MODEL: 'bulbul:v1', // Default TTS model
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
  const lang = SARVAM_CONFIG.LANGUAGES.find(l => l.code === code.split('-')[0]);
  return lang ? lang.name : code;
}

// Get default voice for a language
export function getDefaultVoiceForLanguage(langCode: string): string {
  // Extract base language code (e.g., 'en' from 'en-IN')
  const baseLang = langCode.split('-')[0].toLowerCase();
  
  // Default to 'meera' for all languages as it's a common voice
  const defaultVoices: Record<string, string> = {
    'en': 'meera',
    'hi': 'meera',
    'ta': 'meera',
    'te': 'meera',
    'kn': 'meera',
    'ml': 'meera',
    'bn': 'meera',
    'gu': 'meera',
    'mr': 'meera',
    'pa': 'meera'
  };
  
  return defaultVoices[baseLang] || 'meera';
}
