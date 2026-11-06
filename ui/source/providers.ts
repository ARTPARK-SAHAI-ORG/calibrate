export interface ProviderInfo {
  id: string;
  name: string;
  envVar: string;
  isFilePath?: boolean;
  languages: Set<string>;
}

// ─── TTS Providers ──────────────────────────────────────────
export const TTS_PROVIDERS: ProviderInfo[] = [
  {
    id: 'openai',
    name: 'OpenAI',
    envVar: 'OPENAI_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'marathi', 'tamil',
    ]),
  },
  {
    id: 'google',
    name: 'Google Cloud',
    envVar: 'GOOGLE_APPLICATION_CREDENTIALS',
    isFilePath: true,
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'odia', 'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi',
    ]),
  },
  {
    id: 'elevenlabs',
    name: 'ElevenLabs',
    envVar: 'ELEVENLABS_API_KEY',
    languages: new Set([
      'english', 'hindi', 'sindhi', 'tamil',
    ]),
  },
  {
    id: 'cartesia',
    name: 'Cartesia',
    envVar: 'CARTESIA_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'punjabi', 'tamil', 'telugu', 'gujarati',
    ]),
  },
  {
    id: 'groq',
    name: 'Groq',
    envVar: 'GROQ_API_KEY',
    languages: new Set(['english']),
  },
  {
    id: 'sarvam',
    name: 'Sarvam AI',
    envVar: 'SARVAM_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'odia', 'punjabi', 'tamil', 'telugu', 'gujarati',
    ]),
  },
  {
    id: 'smallest',
    name: 'Smallest AI',
    envVar: 'SMALLEST_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'odia', 'tamil', 'telugu', 'gujarati',
    ]),
  },
];

// ─── STT Providers ──────────────────────────────────────────
export const STT_PROVIDERS: ProviderInfo[] = [
  {
    id: 'deepgram-nova-3',
    name: 'Deepgram',
    envVar: 'DEEPGRAM_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'marathi', 'tamil', 'telugu',
    ]),
  },
  {
    id: 'openai-gpt-4o-transcribe',
    name: 'OpenAI',
    envVar: 'OPENAI_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'marathi', 'tamil',
    ]),
  },
  {
    id: 'groq-whisper-large-v3-turbo',
    name: 'Groq',
    envVar: 'GROQ_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam', 'marathi',
      'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi',
    ]),
  },
  {
    id: 'cartesia-ink-whisper',
    name: 'Cartesia',
    envVar: 'CARTESIA_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi',
    ]),
  },
  {
    id: 'smallest-pulse',
    name: 'Smallest AI',
    envVar: 'SMALLEST_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'odia', 'tamil', 'telugu', 'gujarati',
    ]),
  },
  {
    id: 'google-chirp-3',
    name: 'Google Cloud',
    envVar: 'GOOGLE_APPLICATION_CREDENTIALS',
    isFilePath: true,
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'odia', 'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi',
    ]),
  },
  {
    id: 'sarvam-saaras-v3',
    name: 'Sarvam AI',
    envVar: 'SARVAM_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'odia', 'punjabi', 'tamil', 'telugu', 'gujarati',
    ]),
  },
  {
    id: 'elevenlabs-scribe-v2-realtime',
    name: 'ElevenLabs',
    envVar: 'ELEVENLABS_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'odia', 'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi',
    ]),
  },
  {
    id: 'deepgram-flux',
    name: 'Deepgram Flux',
    envVar: 'DEEPGRAM_API_KEY',
    languages: new Set(['english', 'hindi']),
  },
  {
    id: 'openai-gpt-live-transcribe',
    name: 'OpenAI Live (gpt-live-transcribe)',
    envVar: 'OPENAI_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'marathi', 'tamil',
    ]),
  },
  {
    id: 'openai-gpt-transcribe',
    name: 'OpenAI Transcribe (gpt-transcribe)',
    envVar: 'OPENAI_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam', 'marathi',
      'odia', 'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi', 'maithili',
    ]),
  },
  {
    id: 'sarvam-saaras-v4-realtime',
    name: 'Sarvam AI v4 Realtime',
    envVar: 'SARVAM_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam', 'marathi',
      'odia', 'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi', 'maithili',
    ]),
  },
  {
    id: 'sarvam-saaras-v3-realtime',
    name: 'Sarvam AI v3 Realtime',
    envVar: 'SARVAM_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam', 'marathi',
      'odia', 'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi', 'maithili',
    ]),
  },
  {
    id: 'sarvam-saaras-v4',
    name: 'Sarvam AI v4 (streaming, our end-of-speech detection)',
    envVar: 'SARVAM_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'odia', 'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi',
    ]),
  },
  {
    id: 'cartesia-ink-preview',
    name: 'Cartesia Ink Preview',
    envVar: 'CARTESIA_API_KEY',
    languages: new Set(['english', 'hindi']),
  },
];

export const LANGUAGES = [
  'english', 'hindi', 'kannada', 'bengali', 'malayalam',
  'marathi', 'odia', 'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi',
];

export function getTtsProviderById(id: string): ProviderInfo | undefined {
  return TTS_PROVIDERS.find(p => p.id === id);
}

export function getSttProviderById(id: string): ProviderInfo | undefined {
  return STT_PROVIDERS.find(p => p.id === id);
}

export function getProviderById(id: string, mode: 'tts' | 'stt'): ProviderInfo | undefined {
  return mode === 'tts' ? getTtsProviderById(id) : getSttProviderById(id);
}

export function getTtsProvidersForLanguage(language: string): ProviderInfo[] {
  return TTS_PROVIDERS.filter(p => p.languages.has(language));
}

export function getSttProvidersForLanguage(language: string): ProviderInfo[] {
  return STT_PROVIDERS.filter(p => p.languages.has(language));
}

// The folder a provider's results are stored under. Mirrors stt_output_name in
// calibrate_agent/utils.py: deepgram-flux adds the model it runs for the language.
export function outputDirName(provider: string, mode: 'tts' | 'stt', language: string): string {
  if (mode === 'stt' && provider === 'deepgram-flux') {
    return language === 'english' ? 'deepgram-flux-general-en' : 'deepgram-flux-general-multi';
  }
  return provider;
}

export function getProvidersForLanguage(language: string, mode: 'tts' | 'stt'): ProviderInfo[] {
  return mode === 'tts' ? getTtsProvidersForLanguage(language) : getSttProvidersForLanguage(language);
}

// Keep backwards-compatible aliases
export { getTtsProvidersForLanguage as getProvidersForLanguageTts };
