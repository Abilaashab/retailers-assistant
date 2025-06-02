'use client';

import dynamic from 'next/dynamic';

// Disable SSR for the VoiceInterface component since it uses browser APIs
const VoiceInterface = dynamic(
  () => import('@/components/VoiceInterface'),
  { ssr: false }
);

export default function Home() {
  return <VoiceInterface />;
}
