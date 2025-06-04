'use client';

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { toBcp47Code, SARVAM_CONFIG, VOICES, getDefaultVoiceForLanguage } from '@/config';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  originalContent?: string;
  translatedContent?: string;
  isError?: boolean;
}

const LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'hi', name: 'हिंदी (Hindi)' },
  { code: 'ta', name: 'தமிழ் (Tamil)' },
  { code: 'te', name: 'తెలుగు (Telugu)' },
  { code: 'kn', name: 'ಕನ್ನಡ (Kannada)' },
  { code: 'ml', name: 'മലയാളം (Malayalam)' },
  { code: 'bn', name: 'বাংলা (Bengali)' },
  { code: 'gu', name: 'ગુજરાતી (Gujarati)' },
  { code: 'mr', name: 'मराठी (Marathi)' },
  { code: 'pa', name: 'ਪੰਜਾਬੀ (Punjabi)' },
];

export default function VoiceInterface() {
  const [selectedLanguage, setSelectedLanguage] = useState('en-IN');
  const [selectedVoice, setSelectedVoice] = useState(getDefaultVoiceForLanguage('en-IN'));
  const [isListening, setIsListening] = useState(false);
  
  // Get available voices for the selected language
  const getAvailableVoices = useCallback((lang: string) => {
    const baseLang = lang.split('-')[0].toLowerCase();
    
    // Combine all available voices from VOICES constant
    const allVoices = [
      ...VOICES.female.map(v => ({ ...v, gender: 'female' })),
      ...VOICES.male.map(v => ({ ...v, gender: 'male' }))
    ];
    
    // Return all available voices for all languages
    return allVoices;
  }, []);
  
  // Update selected voice when language changes
  useEffect(() => {
    const defaultVoice = getDefaultVoiceForLanguage(selectedLanguage);
    console.log(`Language changed to ${selectedLanguage}, setting default voice to ${defaultVoice}`);
    setSelectedVoice(defaultVoice);
  }, [selectedLanguage, getDefaultVoiceForLanguage]);
  
  // Get available voices for the current language
  const availableVoices = useMemo(() => {
    const baseLang = selectedLanguage.split('-')[0].toLowerCase();
    const voices = getAvailableVoices(selectedLanguage);
    console.log(`Available voices for ${baseLang}:`, voices);
    return voices;
  }, [selectedLanguage, getAvailableVoices]);
  const [conversation, setConversation] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const lastAssistantMessage = conversation.slice().reverse().find(msg => msg.role === 'assistant');
  
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition,
    isMicrophoneAvailable
  } = useSpeechRecognition();

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const form = e.target as HTMLFormElement;
    const input = form.elements.namedItem('textInput') as HTMLInputElement;
    const text = input.value.trim();
    
    if (text) {
      handleSubmit(text);
      input.value = '';
    }
  };
  
  const toggleListening = () => {
    if (isListening) {
      SpeechRecognition.stopListening();
      setIsListening(false);
    } else {
      resetTranscript();
      SpeechRecognition.startListening({
        language: selectedLanguage,
      });
      setIsListening(true);
    }
  };
  
  const handleSubmit = async (text: string, audioData?: string) => {
    if ((!text && !audioData) || isProcessing) return;

    try {
      setIsProcessing(true);
      
      // Add user's message to conversation
      if (text) {
        setConversation(prev => [...prev, { role: 'user', content: text }]);
      } else if (audioData) {
        setConversation(prev => [...prev, { role: 'user', content: '🎤 [Voice message]' }]);
      }

      const requestBody = {
        text,
        sourceLang: selectedLanguage,
        targetLang: selectedLanguage,
        isAudio: !!audioData,
        audioData,
        audioFormat: 'audio/webm',
        skipTranslation: selectedLanguage.startsWith('en')
      };

      console.log('Sending request to chat API:', JSON.stringify({
        ...requestBody,
        audioData: audioData ? '[audio data]' : undefined
      }, null, 2));

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Chat API error:', errorText);
        throw new Error(`Failed to get response from server: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Chat API response:', data);
      
      // Update user message with translated content if available
      if (data.translated_text) {
        setConversation(prev => {
          const updated = [...prev];
          const lastUserMessage = updated[updated.length - 1];
          if (lastUserMessage.role === 'user') {
            lastUserMessage.translatedContent = data.translated_text;
          }
          return updated;
        });
      }

      // Add assistant's response
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.agent_response || data.error || 'No response from assistant',
      };
      
      setConversation(prev => [...prev, assistantMessage]);
      
      // Speak the assistant's response
      if (assistantMessage.content) {
        speakText(assistantMessage.content, selectedLanguage);
      }
    } catch (error) {
      console.error('Error:', error);
      setConversation(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        isError: true
      }]);
    } finally {
      setIsProcessing(false);
    }
  };
  
  const startAudioRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks.current = [];
      
      mediaRecorder.current = new MediaRecorder(stream);
      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };
      
      mediaRecorder.current.onstop = () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = () => {
          const base64data = reader.result as string;
          handleSubmit('', base64data);
        };
      };
      
      mediaRecorder.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting audio recording:', error);
      const errorMessage = 'Could not access microphone. Please ensure you have granted microphone permissions.';
      setConversation(prev => [...prev, { role: 'assistant', content: errorMessage, isError: true }]);
    }
  };
  
  const stopAudioRecording = () => {
    if (mediaRecorder.current && mediaRecorder.current.state !== 'inactive') {
      mediaRecorder.current.stop();
      mediaRecorder.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };
  
  const speakText = async (text: string, lang: string) => {
    if (!text) return;

    try {
      console.log('Starting text-to-speech for text:', text);
      setIsSpeaking(true);
      
      // Stop any currently playing audio
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }

      console.log('Sending TTS request with voice:', selectedVoice);
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          language_code: lang || 'en-IN',
          voice: selectedVoice,  // Changed from voice_id to voice
          sample_rate: 24000, // Standard sample rate for TTS
        }),
      });

      console.log('TTS Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Failed to read error response');
        console.error('TTS API Error:', {
          status: response.status,
          statusText: response.statusText,
          error: errorText,
        });
        throw new Error(`Failed to convert text to speech: ${response.statusText}`);
      }

      // Handle the TTS response
      const jsonResponse = await response.json();
      console.log('TTS Response:', jsonResponse);
      
      if (!jsonResponse.audio) {
        throw new Error('No audio data in response');
      }
      
      // Convert base64 to Blob
      const base64Data = jsonResponse.audio.split(';base64,').pop() || jsonResponse.audio;
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const audioBlob = new Blob([byteArray], { type: 'audio/wav' });

      const audioUrl = URL.createObjectURL(audioBlob);
      console.log('Created audio URL from blob');
      
      // Create and play audio
      const audio = new Audio(audioUrl);
      currentAudioRef.current = audio;
      
      audio.onended = () => {
        console.log('Audio playback ended');
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
      };
      
      audio.onerror = (error) => {
        console.error('Audio playback error:', error);
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
      };
      
      console.log('Starting audio playback');
      await audio.play().catch(error => {
        console.error('Error playing audio:', error);
        throw error;
      });
      
    } catch (error) {
      console.error('Error in speakText:', error);
      setIsSpeaking(false);
      // Re-throw to allow caller to handle the error
      throw error;
      setIsSpeaking(false);
    }
  };

  const stopSpeaking = () => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
      setIsSpeaking(false);
    }
  };
  
  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-center text-gray-900 mb-8">Voice Assistant</h1>
        
        {/* Language and Voice Selection */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div>
            <label htmlFor="language" className="block text-sm font-medium text-gray-700 mb-1">
              Select Language
            </label>
            <select
              id="language"
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 text-gray-900 bg-white"
              disabled={isListening || isProcessing}
            >
              {LANGUAGES.map((lang) => (
                <option key={lang.code} value={toBcp47Code(lang.code)}>
                  {lang.name}
                </option>
              ))}
            </select>
          </div>
          
          <div>
            <label htmlFor="voice" className="block text-sm font-medium text-gray-700 mb-1">
              Select Voice
            </label>
            <select
              id="voice"
              value={selectedVoice}
              onChange={(e) => {
                console.log('Selected voice changed to:', e.target.value);
                setSelectedVoice(e.target.value);
              }}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 text-gray-900 bg-white"
              disabled={isProcessing}
            >
              <optgroup label="Available Voices">
                {availableVoices.map((voice) => (
                  <option key={voice.id} value={voice.id}>
                    {voice.name} ({voice.gender.charAt(0).toUpperCase() + voice.gender.slice(1)})
                  </option>
                ))}
              </optgroup>
            </select>
          </div>
        </div>
        
        {/* Conversation */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6 h-[50vh] overflow-y-auto">
          {conversation.length === 0 ? (
            <div className="h-full flex items-center justify-center text-gray-500">
              Start a conversation by clicking the microphone or typing a message
            </div>
          ) : (
            <div className="space-y-4">
              {conversation.map((message, index) => (
                <div 
                  key={index}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div 
                    className={`max-w-[80%] rounded-lg p-4 ${
                      message.role === 'user'
                        ? 'bg-indigo-600 text-white rounded-br-none'
                        : message.isError
                        ? 'bg-red-100 text-red-800 rounded-bl-none'
                        : 'bg-gray-100 text-gray-800 rounded-bl-none'
                    }`}
                  >
                    <div className="text-sm font-medium mb-1">
                      {message.role === 'user' ? 'You' : 'Xenie'}
                    </div>
                    
                    {/* For user messages, show original text first, then translated text */}
                    {message.role === 'user' && (
                      <>
                        <div className="whitespace-pre-wrap mb-2">
                          {message.originalContent || message.content}
                        </div>
                        
                        {message.translatedContent && message.translatedContent !== (message.originalContent || message.content) && (
                          <div className="text-sm italic border-t border-white/20 pt-2 mt-2">
                            <span className="opacity-70">(Translated) </span>
                            {message.translatedContent}
                          </div>
                        )}
                      </>
                    )}
                    
                    {/* For assistant messages, just show the content */}
                    {message.role === 'assistant' && (
                      <div className="whitespace-pre-wrap">
                        {message.content}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {isProcessing && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 text-gray-800 rounded-lg rounded-bl-none p-4 max-w-[80%]">
                    <div className="flex space-x-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
        
        {/* Input Area */}
        <div className="bg-white rounded-lg shadow-md p-4">
          <form onSubmit={handleTextSubmit} className="flex space-x-2">
            <input
              type="text"
              name="textInput"
              placeholder="Type your message..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              disabled={isProcessing || isListening || isRecording}
            />
            <button
              type="submit"
              className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
              disabled={isProcessing || isListening || isRecording}
            >
              Send
            </button>
          </form>
          
          <div className="mt-4 flex justify-center">
            <button
              type="button"
              onClick={toggleListening}
              disabled={!isMicrophoneAvailable || isProcessing || isRecording}
              className={`flex items-center justify-center w-12 h-12 rounded-full ${
                isListening ? 'bg-red-500' : 'bg-indigo-600'
              } text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors`}
              aria-label={isListening ? 'Stop listening' : 'Start listening'}
            >
              {isListening ? (
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              )}
            </button>
            
            {!isMicrophoneAvailable && (
              <p className="ml-2 text-sm text-red-600">
                Microphone access is required for voice input
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
