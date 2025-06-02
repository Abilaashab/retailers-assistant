'use client';

import { useState, useEffect, useRef } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { toBcp47Code } from '@/config';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  isError?: boolean;
};

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
  const [isListening, setIsListening] = useState(false);
  const [conversation, setConversation] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);

  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition,
    isMicrophoneAvailable
  } = useSpeechRecognition();

  useEffect(() => {
    if (!browserSupportsSpeechRecognition) {
      alert("Your browser doesn't support speech recognition. Please use a modern browser like Chrome or Edge.");
    }
  }, [browserSupportsSpeechRecognition]);

  useEffect(() => {
    setIsListening(listening);
  }, [listening]);

  const handleLanguageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedLanguage(e.target.value);
  };

  const toggleListening = async () => {
    if (isListening) {
      SpeechRecognition.stopListening();
      if (transcript) {
        await handleSubmit(transcript);
      }
      stopAudioRecording();
    } else {
      resetTranscript();
      await startAudioRecording();
      SpeechRecognition.startListening({ 
        language: toBcp47Code(selectedLanguage),
        continuous: true,
        interimResults: true
      });
    }
  };

  const startAudioRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];
      
      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };
      
      mediaRecorder.current.onstop = async () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Convert blob to base64 for the API
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

  const handleSubmit = async (text: string, audioData?: string) => {
    const isAudio = !!audioData;
    if (!text.trim() && !audioData) return;

    console.log('Submitting:', isAudio ? 'Audio' : 'Text');
    console.log('Selected language:', selectedLanguage);

    // Add user message to conversation
    const userMessage: Message = { 
      role: 'user', 
      content: isAudio ? '[Audio message]' : text 
    };
    
    setConversation(prev => [...prev, userMessage]);
    setIsProcessing(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: isAudio ? '' : text,
          sourceLang: selectedLanguage,
          targetLang: 'en', // Always translate to English for the supervisor agent
          isAudio,
          audioData,
          audioFormat: 'audio/wav'
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('API Response:', data);
      
      // Add bot response to conversation
      const botResponse: Message = {
        role: 'assistant',
        content: data.translated_text || 'No response content'
      };
      
      setConversation(prev => [...prev, botResponse]);
      
      // Play TTS if available
      if (data.audioUrl) {
        const audio = new Audio(data.audioUrl);
        audio.play().catch(e => console.error('Error playing audio:', e));
      }
      
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = error instanceof Error 
        ? `Error: ${error.message}` 
        : 'An unknown error occurred';
      
      setConversation(prev => [...prev, { 
        role: 'assistant', 
        content: errorMessage, 
        isError: true 
      }]);
    } finally {
      setIsProcessing(false);
      resetTranscript();
    }
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const input = (e.target as any).elements.textInput;
    if (input.value.trim()) {
      handleSubmit(input.value);
      input.value = '';
    }
  };

  // Format language options for display
  const languageOptions = LANGUAGES.map(lang => ({
    ...lang,
    displayName: lang.name
  }));

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Voice Assistant</h1>
          <p className="text-gray-600">Speak or type in your preferred language</p>
        </div>

        {/* Conversation History */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6 max-h-[60vh] overflow-y-auto">
          {conversation.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              <p>Start a conversation by speaking or typing below</p>
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
                      {message.role === 'user' ? 'You' : 'Assistant'}
                    </div>
                    <div className="whitespace-pre-wrap">{message.content}</div>
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

        {/* Controls */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="mb-4">
            <label htmlFor="language" className="block text-sm font-medium text-gray-800 mb-1">
              Select Language
            </label>
            <select
              id="language"
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 text-gray-900 bg-white"
              disabled={isListening || isProcessing}
            >
              {languageOptions.map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.displayName}
                </option>
              ))}
            </select>
          </div>

          {/* Recording Controls */}
          <div className="flex flex-col items-center mb-6">
            <div className="flex items-center justify-center w-full mb-4">
              <div className="flex-1 flex items-center justify-end pr-4">
                {isRecording && (
                  <div className="flex items-center">
                    <div className="h-3 w-3 rounded-full bg-red-500 animate-pulse mr-2"></div>
                    <span className="text-sm text-gray-600">Recording...</span>
                  </div>
                )}
              </div>
              
              <button
                type="button"
                onClick={toggleListening}
                disabled={!isMicrophoneAvailable || isProcessing}
                className={`flex-shrink-0 flex items-center justify-center w-16 h-16 rounded-full ${
                  isListening || isRecording
                    ? 'bg-red-500 hover:bg-red-600'
                    : 'bg-indigo-600 hover:bg-indigo-700'
                } text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors ${
                  (!isMicrophoneAvailable || isProcessing) ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                aria-label={isListening || isRecording ? 'Stop recording' : 'Start speaking'}
              >
                {isListening || isRecording ? (
                  <svg
                    className="h-8 w-8"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"
                    />
                  </svg>
                ) : (
                  <svg
                    className="h-8 w-8"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                    />
                  </svg>
                )}
              </button>

              <div className="flex-1 pl-4">
                <div className="text-sm text-gray-600">
                  {isListening || isRecording ? (
                    <span className="flex items-center">
                      <span className="flex h-3 w-3 mr-2">
                        <span className="animate-ping absolute inline-flex h-3 w-3 rounded-full bg-red-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                      </span>
                      {isRecording ? 'Recording...' : 'Listening...'}
                    </span>
                  ) : (
                    <span className={!isMicrophoneAvailable || isProcessing ? 'text-gray-400' : ''}>
                      {!isMicrophoneAvailable 
                        ? 'Microphone not available' 
                        : isProcessing 
                          ? 'Processing...' 
                          : 'Click to speak'}
                    </span>
                  )}
                </div>
                {transcript && (
                  <div className="mt-2 text-sm text-gray-500">
                    <span className="font-medium">You said:</span> {transcript}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Text Input Form */}
          <form onSubmit={handleTextSubmit} className="w-full mt-4">
            <div className="flex">
              <input
                type="text"
                name="textInput"
                placeholder="Or type your message here..."
                className={`flex-1 px-4 py-2 border ${
                  isProcessing || isListening || isRecording
                    ? 'border-gray-200 bg-gray-50 cursor-not-allowed'
                    : 'border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-transparent'
                } rounded-l-md focus:outline-none transition-colors`}
                disabled={isProcessing || isListening || isRecording}
              />
              <button
                type="submit"
                className={`px-4 py-2 ${
                  isProcessing || isListening || isRecording
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-indigo-600 hover:bg-indigo-700 focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'
                } text-white rounded-r-md focus:outline-none transition-colors`}
                disabled={isProcessing || isListening || isRecording}
              >
                {isProcessing ? 'Sending...' : 'Send'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
