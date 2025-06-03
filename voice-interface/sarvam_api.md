# Sarvam AI API Integration

This document outlines how the Sarvam AI API is integrated and used within the application.

## Base Configuration

- **Base URL**: `https://api.sarvam.ai`
- **Authentication**: API Key passed in the `api-subscription-key` header
- **Environment Variable**: `SARVAM_AI_API`

## API Endpoints

### 1. Text-to-Text Translation

**Endpoint**: `POST /v1/translate`

**Purpose**: Translates text from one language to another.

**Request Headers**:
```
Content-Type: application/json
api-subscription-key: <your_api_key>
```

**Request Body (JSON)**:
```json
{
  "input": "text to translate",
  "source_language_code": "en",
  "target_language_code": "hi",
  "speaker_gender": "male/female",
  "mode": "formal",
  "enable_preprocessing": true,
  "numerals_format": "international"
}
```

**Response**:
```json
{
  "translated_text": "translated text",
  "source_language_code": "en",
  "target_language_code": "hi",
  "request_id": "unique-request-id"
}
```

### 2. Speech-to-Text with Translation

**Endpoint**: `POST /v1/transcribe/translate`

**Purpose**: Transcribes speech to text and optionally translates it to another language.

**Request Headers**:
```
api-subscription-key: <your_api_key>
```

**Request Body (multipart/form-data)**:
- `file`: Audio file (WAV format)
- `source_language`: Source language code (e.g., "hi-IN")
- `target_language`: Target language code (e.g., "en-IN")
- `speaker_gender`: "male" or "female"
- `mode`: "formal" or "informal"
- `enable_preprocessing`: "true" or "false"

**Response**:
```json
{
  "transcript": "transcribed text",
  "translated_text": "translated text",
  "language_code": "source-language-code",
  "request_id": "unique-request-id"
}
```

### 3. Text-to-Speech

**Endpoint**: `POST /v1/tts`

**Purpose**: Converts text to speech.

**Request Headers**:
```
api-subscription-key: <your_api_key>
```

**Request Body (JSON)**:
```json
{
  "text": "text to speak",
  "language_code": "en-IN",
  "voice_id": "default",
  "sample_rate": 24000
}
```

**Response**:
- Audio file in the specified format

## Supported Languages

The application supports the following languages:

| Code | Language          | Native Name              |
|------|-------------------|--------------------------|
| en   | English          | English                 |
| hi   | Hindi            | हिंदी                   |
| mr   | Marathi          | मराठी                   |
| ta   | Tamil            | தமிழ்                   |
| te   | Telugu           | తెలుగు                  |
| kn   | Kannada          | ಕನ್ನಡ                   |
| ml   | Malayalam        | മലയാളം                  |
| bn   | Bengali          | বাংলা                    |
| gu   | Gujarati         | ગુજરાતી                  |
| pa   | Punjabi          | ਪੰਜਾਬੀ                   |
| or   | Odia             | ଓଡ଼ିଆ                    |
| as   | Assamese         | অসমীয়া                  |

## Implementation Details

### Environment Variables

```bash
# Required
SARVAM_AI_API=your_api_key_here
```

### Configuration

All API-related configuration is stored in `src/config.ts` and includes:
- Base URL
- API endpoints
- Supported languages
- Default settings (sample rate, voice ID, etc.)

### Key Components

1. **API Routes** (`src/app/api/chat/route.ts`):
   - Handles incoming chat requests
   - Routes requests to appropriate Sarvam API endpoints
   - Manages error handling and response formatting

2. **Configuration** (`src/config.ts`):
   - Centralized configuration for all Sarvam API endpoints
   - Language code mappings and utilities

## Error Handling

The application includes comprehensive error handling for API responses, including:
- Invalid API keys (403 errors)
- Rate limiting (429 errors)
- Invalid language codes (400 errors)
- Server errors (5xx)

## Rate Limits

Please refer to the official Sarvam AI documentation for current rate limits and quotas.

## Testing

To test the integration:
1. Set the `SARVAM_AI_API` environment variable
2. Start the development server
3. Use the application interface to send text or audio messages
4. Monitor the console for API requests and responses

## Troubleshooting

Common issues and solutions:

1. **403 Forbidden**: Check that your API key is valid and properly set in the environment variables
2. **400 Bad Request**: Verify that all required parameters are included and in the correct format
3. **429 Too Many Requests**: You've exceeded the rate limit. Wait before making more requests
4. **500 Internal Server Error**: Contact Sarvam AI support if the issue persists

## Dependencies

- Node.js
- Next.js API routes
- Fetch API for HTTP requests

## Security Considerations

- API keys should never be exposed in client-side code
- All API calls are made server-side through Next.js API routes
- Environment variables are used to store sensitive information

## References

- [Sarvam AI Official Documentation](https://docs.sarvam.ai/)
- [Next.js API Routes](https://nextjs.org/docs/api-routes/introduction)
