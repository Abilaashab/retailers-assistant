// Sarvam AI API configuration
export const SARVAM_CONFIG = {
  // API base URL
  BASE_URL: 'https://api.sarvam.ai',
  
  // API endpoints
  ENDPOINTS: {
    SPEECH_TO_TEXT_TRANSLATE: '/speech-to-text-translate',
    TEXT_TO_TEXT_TRANSLATE: '/text-to-text-translate',
    TEXT_TO_SPEECH: '/text-to-speech'
  },
  
  // API key header name
  API_KEY_HEADER: 'api-subscription-key',
  
  // Supported languages with their BCP-47 codes
  LANGUAGES: [
    { code: 'en-IN', name: 'English (India)' },
    { code: 'hi-IN', name: 'हिंदी (Hindi)' },
    { code: 'ta-IN', name: 'தமிழ் (Tamil)' },
    { code: 'te-IN', name: 'తెలుగు (Telugu)' },
    { code: 'kn-IN', name: 'ಕನ್ನಡ (Kannada)' },
    { code: 'ml-IN', name: 'മലയാളം (Malayalam)' },
    { code: 'bn-IN', name: 'বাংলা (Bengali)' },
    { code: 'gu-IN', name: 'ગુજરાતી (Gujarati)' },
    { code: 'mr-IN', name: 'मराठी (Marathi)' },
    { code: 'pa-IN', name: 'ਪੰਜਾਬੀ (Punjabi)' },
  ]
};

// Helper function to get language name from code
export function getLanguageName(code: string): string {
  // Handle both short codes (en) and BCP-47 codes (en-IN)
  const shortCode = code.split('-')[0];
  const lang = SARVAM_CONFIG.LANGUAGES.find(lang => 
    lang.code === code || lang.code.startsWith(`${shortCode}-`)
  );
  return lang ? lang.name : code;
}

// Helper function to map language code to BCP-47 format
export function toBcp47Code(code: string): string {
  // If it's already in BCP-47 format, return as is
  if (code.includes('-')) return code;
  
  // Try to find a matching BCP-47 code
  const lang = SARVAM_CONFIG.LANGUAGES.find(lang => 
    lang.code.startsWith(`${code}-`)
  );
  
  // Default to IN country code if not found
  return lang ? lang.code : `${code}-IN`;
};
