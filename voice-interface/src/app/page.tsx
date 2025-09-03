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
              Use the voice interface below to interact with the agent.
            </p>
            
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
