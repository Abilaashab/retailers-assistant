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
    <div className="min-h-screen bg-white">
      <VoiceInterface />
    </div>
  );
}
