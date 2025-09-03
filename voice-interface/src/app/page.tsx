'use client';

import dynamic from 'next/dynamic';
import Navigation from '@/components/Navigation';

// Disable SSR for the VoiceInterface component since it uses browser APIs
const VoiceInterface = dynamic(
  () => import('@/components/VoiceInterface'),
  { ssr: false }
);

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navigation />
      <main className="flex-grow">
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="px-4 py-6 sm:px-0">
            <h1 className="text-3xl font-bold text-gray-900 mb-6">Welcome to Voice Agent</h1>
            <p className="text-lg text-gray-600 mb-8">
              Experience real-time text-to-speech with our streaming TTS technology. 
              Try out the new streaming TTS feature for a more responsive experience.
            </p>
            
            <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6 mb-8">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Features</h2>
              <ul className="list-disc pl-5 space-y-2 text-gray-700">
                <li>Real-time text-to-speech conversion</li>
                <li>Multiple voices and languages</li>
                <li>Streaming audio for instant playback</li>
                <li>Responsive design for all devices</li>
              </ul>
              
              <div className="mt-6">
                <a
                  href="/tts-stream"
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                >
                  Try Streaming TTS
                </a>
              </div>
            </div>
            
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Voice Interface</h2>
            <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
              <VoiceInterface />
            </div>
          </div>
        </div>
      </main>
      
      <footer className="bg-white border-t border-gray-200">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <p className="text-center text-gray-500 text-sm">
            &copy; {new Date().getFullYear()} Voice Agent. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}
