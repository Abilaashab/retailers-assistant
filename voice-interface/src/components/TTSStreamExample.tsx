'use client';

import { useState } from 'react';
import { useTTSStream } from '@/hooks/useTTSStream';

export default function TTSStreamExample() {
  const [text, setText] = useState('Hello, this is a test of the streaming TTS functionality.');
  const [language, setLanguage] = useState('en-IN');
  const [voice, setVoice] = useState('meera');
  
  const { speak, stop, isLoading, isPlaying, error } = useTTSStream();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (isLoading || isPlaying) {
      stop();
      return;
    }
    
    speak(text, {
      language_code: language,
      voice,
      onAudioChunk: (chunk) => {
        console.log('Received audio chunk:', chunk.byteLength, 'bytes');
      },
      onComplete: () => {
        console.log('TTS streaming completed');
      },
      onError: (err) => {
        console.error('TTS error:', err);
      },
    });
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Streaming TTS Example</h1>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          Error: {error.message}
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="text" className="block text-sm font-medium text-gray-700 mb-1">
            Text to speak
          </label>
          <textarea
            id="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            rows={4}
            required
          />
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label htmlFor="language" className="block text-sm font-medium text-gray-700 mb-1">
              Language
            </label>
            <select
              id="language"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="en-IN">English (India)</option>
              <option value="hi-IN">Hindi</option>
              <option value="ta-IN">Tamil</option>
              <option value="te-IN">Telugu</option>
              <option value="kn-IN">Kannada</option>
              <option value="ml-IN">Malayalam</option>
            </select>
          </div>
          
          <div>
            <label htmlFor="voice" className="block text-sm font-medium text-gray-700 mb-1">
              Voice
            </label>
            <select
              id="voice"
              value={voice}
              onChange={(e) => setVoice(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <optgroup label="Female Voices">
                <option value="diya">Diya</option>
                <option value="maya">Maya</option>
                <option value="meera">Meera</option>
                <option value="pavithra">Pavithra</option>
                <option value="maitreyi">Maitreyi</option>
                <option value="misha">Misha</option>
              </optgroup>
              <optgroup label="Male Voices">
                <option value="amol">Amol</option>
                <option value="arjun">Arjun</option>
                <option value="amartya">Amartya</option>
                <option value="arvind">Arvind</option>
                <option value="neel">Neel</option>
                <option value="vian">Vian</option>
              </optgroup>
            </select>
          </div>
        </div>
        
        <div className="pt-2">
          <button
            type="submit"
            className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
              isLoading || isPlaying
                ? 'bg-red-600 hover:bg-red-700 focus:ring-red-500'
                : 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500'
            } focus:outline-none focus:ring-2 focus:ring-offset-2`}
            disabled={!text.trim()}
          >
            {isLoading ? (
              'Processing...'
            ) : isPlaying ? (
              'Stop Playing'
            ) : (
              'Speak'
            )}
          </button>
        </div>
      </form>
      
      <div className="mt-8 p-4 bg-gray-50 rounded-md">
        <h2 className="text-lg font-medium mb-2">How it works:</h2>
        <ul className="list-disc pl-5 space-y-1 text-sm text-gray-600">
          <li>Enter the text you want to convert to speech</li>
          <li>Select the language and voice</li>
          <li>Click "Speak" to start streaming the audio</li>
          <li>Click "Stop Playing" to stop the audio at any time</li>
        </ul>
        
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded text-sm text-blue-700">
          <p className="font-medium">Note:</p>
          <p className="mt-1">
            This uses WebSockets to stream the audio in real-time, which provides a better user experience 
            for longer texts as the audio starts playing before the entire text is processed.
          </p>
        </div>
      </div>
    </div>
  );
}
