'use client';

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { toBcp47Code, SARVAM_CONFIG, VOICES, getDefaultVoiceForLanguage } from '@/config';
import styles from './VoiceInterface.module.css';

// Icons
const MicrophoneIcon = ({ isListening }: { isListening: boolean }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    {isListening ? (
      <>
        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        <line x1="12" y1="19" x2="12" y2="23" />
        <line x1="8" y1="23" x2="16" y2="23" />
      </>
    ) : (
      <>
        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        <line x1="12" y1="19" x2="12" y2="23" />
        <line x1="8" y1="23" x2="16" y2="23" />
      </>
    )}
  </svg>
);

const SendIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const MicOffIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <line x1="1" y1="1" x2="23" y2="23" />
    <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6" />
    <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23" />
    <line x1="12" y1="19" x2="12" y2="23" />
    <line x1="8" y1="23" x2="16" y2="23" />
  </svg>
);

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

  // Add a ref to store the latest transcript
  const transcriptRef = useRef('');
  
  // Update the ref whenever transcript changes
  useEffect(() => {
    transcriptRef.current = transcript;
  }, [transcript]);

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
  
  const toggleListening = async () => {
    if (isListening) {
      console.log('Stopping voice recognition...');
      SpeechRecognition.stopListening();
      setIsListening(false);
      console.log('Voice recognition stopped');
      
      // Only submit if there's actual text in the transcript
      if (transcriptRef.current.trim()) {
        console.log('Submitting transcript:', transcriptRef.current);
        await handleSubmit(transcriptRef.current);
        resetTranscript();
      }
    } else {
      console.log('Starting voice recognition with language:', selectedLanguage);
      try {
        resetTranscript();
        await SpeechRecognition.startListening({
          language: selectedLanguage,
          continuous: false, // Changed to false to auto-stop after speech ends
        });
        setIsListening(true);
        console.log('Voice recognition started');
      } catch (error) {
        console.error('Error starting voice recognition:', error);
        setConversation(prev => [...prev, { 
          role: 'assistant', 
          content: 'Error starting voice recognition. Please check your microphone permissions.', 
          isError: true 
        }]);
      }
    }
  };
  
  useEffect(() => {
    if (transcript) {
      console.log('Transcript updated:', transcript);
    }
  }, [transcript]);
  
  useEffect(() => {
    console.log('Recording state changed - isRecording:', isRecording, 'isListening:', isListening);
  }, [isRecording, isListening]);
  
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
    console.log('Starting audio recording...');
    try {
      console.log('Requesting microphone access...');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log('Microphone access granted, initializing recorder...');
      
      audioChunks.current = [];
      
      mediaRecorder.current = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      console.log('MediaRecorder created with state:', mediaRecorder.current.state);
      
      mediaRecorder.current.ondataavailable = (event) => {
        console.log('Data available event, size:', event.data.size);
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };
      
      mediaRecorder.current.onstop = async () => {
        console.log('MediaRecorder stopped, processing audio...');
        try {
          console.log('Audio chunks count:', audioChunks.current.length);
          const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' });
          console.log('Audio blob created, size:', audioBlob.size);
          
          const reader = new FileReader();
          
          reader.onloadend = () => {
            console.log('FileReader loaded audio data');
            const base64data = reader.result as string;
            console.log('Audio data length:', base64data.length);
            handleSubmit('', base64data);
          };
          
          reader.onerror = (error) => {
            console.error('Error reading audio data:', error);
            setConversation(prev => [...prev, { 
              role: 'assistant', 
              content: 'Error processing the audio. Please try again.', 
              isError: true 
            }]);
          };
          
          console.log('Starting to read audio blob as data URL...');
          reader.readAsDataURL(audioBlob);
        } catch (error) {
          console.error('Error in onstop handler:', error);
          setConversation(prev => [...prev, { 
            role: 'assistant', 
            content: 'Error processing the audio. Please try again.', 
            isError: true 
          }]);
        }
      };
      
      mediaRecorder.current.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        setConversation(prev => [...prev, { 
          role: 'assistant', 
          content: 'Error recording audio. Please try again.', 
          isError: true 
        }]);
      };
      
      console.log('Starting MediaRecorder...');
      mediaRecorder.current.start(100);
      console.log('MediaRecorder started, state:', mediaRecorder.current.state);
      setIsRecording(true);
      console.log('Audio recording started');
    } catch (error) {
      console.error('Error in startAudioRecording:', error);
      const errorMessage = 'Could not access microphone. Please ensure you have granted microphone permissions.';
      setConversation(prev => [...prev, { role: 'assistant', content: errorMessage, isError: true }]);
    }
  };
  
  const stopAudioRecording = () => {
    if (mediaRecorder.current && mediaRecorder.current.state !== 'inactive') {
      mediaRecorder.current.stop();
      mediaRecorder.current.stream.getTracks().forEach(track => {
        track.stop();
        track.enabled = false;
      });
      setIsRecording(false);
      console.log('Stopped audio recording');
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
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerContent}>
          <h1 className={styles.title}>Mitra</h1>
          <p className={styles.subtitle}>Ram General store's voice assistant</p>
        </div>
        <div className={styles.controls}>
          <select
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
            className={styles.select}
            aria-label="Select language"
          >
            {LANGUAGES.map((lang) => (
              <option key={lang.code} value={toBcp47Code(lang.code)}>
                {lang.name}
              </option>
            ))}
          </select>
          <select
            value={selectedVoice}
            onChange={(e) => setSelectedVoice(e.target.value)}
            className={styles.select}
            disabled={availableVoices.length === 0}
            aria-label="Select voice"
          >
            {availableVoices.map((voice) => (
              <option key={voice.id} value={voice.id}>
                {voice.name} ({voice.gender})
              </option>
            ))}
          </select>
        </div>
      </header>

      <div className={styles.chatContainer}>
        {conversation.length === 0 ? (
          <div className={styles.emptyState}>
            <button
              onClick={toggleListening}
              disabled={!browserSupportsSpeechRecognition || !isMicrophoneAvailable}
              className={`${styles.microphoneButton} ${isListening ? styles.listening : ''}`}
              aria-label={isListening ? 'Stop listening' : 'Start speaking'}
            >
              {!browserSupportsSpeechRecognition || !isMicrophoneAvailable ? (
                <MicOffIcon />
              ) : (
                <MicrophoneIcon isListening={isListening} />
              )}
            </button>
            <p className={styles.emptyText}>
              {!browserSupportsSpeechRecognition
                ? "Your browser doesn't support speech recognition."
                : !isMicrophoneAvailable
                ? 'Microphone access is required for voice input.'
                : 'Start a conversation by clicking the microphone or typing a message'}
            </p>
          </div>
        ) : (
          <div className={styles.messages}>
            {conversation.map((message, index) => (
              <div
                key={index}
                className={`${styles.message} ${
                  message.role === 'user' ? styles.userMessage : styles.assistantMessage
                }`}
              >
                <div className="whitespace-pre-wrap">
                  {message.content}
                  {message.originalContent && message.originalContent !== message.content && (
                    <>
                      <hr className="my-2 border-t border-opacity-20" />
                      <span className="text-xs opacity-70">{message.originalContent}</span>
                    </>
                  )}
                </div>
              </div>
            ))}

            {isProcessing && (
              <div className={styles.typingIndicator}>
                <div className={styles.typingDot}></div>
                <div className={styles.typingDot}></div>
                <div className={styles.typingDot}></div>
              </div>
            )}
          </div>
        )}

        <div className={styles.inputContainer}>
          <form onSubmit={handleTextSubmit} className={styles.inputWrapper}>
            <input
              type="text"
              name="textInput"
              placeholder="Type your message..."
              className={styles.input}
              disabled={isProcessing || isListening || isRecording}
              aria-label="Type your message"
            />
            <button
              type="submit"
              className={styles.sendButton}
              disabled={isProcessing}
              aria-label="Send message"
            >
              <SendIcon />
            </button>
            <button
              type="button"
              onClick={toggleListening}
              disabled={!browserSupportsSpeechRecognition || !isMicrophoneAvailable || isProcessing}
              className={`${styles.micButton} ${isListening ? styles.listening : ''}`}
              aria-label={isListening ? 'Stop listening' : 'Start voice input'}
            >
              {!browserSupportsSpeechRecognition || !isMicrophoneAvailable ? (
                <MicOffIcon />
              ) : (
                <MicrophoneIcon isListening={isListening} />
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
