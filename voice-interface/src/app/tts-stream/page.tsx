'use client';

import dynamic from 'next/dynamic';
import Navigation from '@/components/Navigation';

// Dynamically import the TTSStreamExample component with no SSR
const TTSStreamExample = dynamic(
  () => import('@/components/TTSStreamExample'),
  { ssr: false }
);

export default function TTSStreamPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navigation />
      <main className="flex-grow py-12 px-4 sm:px-6 lg:px-8 bg-gray-50">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Streaming TTS Demo</h1>
            <p className="text-lg text-gray-600">
              Experience real-time text-to-speech with streaming audio playback
            </p>
          </div>
          <div className="bg-white shadow-lg rounded-lg p-6">
            <TTSStreamExample />
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
