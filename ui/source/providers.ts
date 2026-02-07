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
    id: 'deepgram',
    name: 'Deepgram',
    envVar: 'DEEPGRAM_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'marathi', 'tamil', 'telugu',
    ]),
  },
  {
    id: 'openai',
    name: 'OpenAI',
    envVar: 'OPENAI_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'marathi', 'tamil',
    ]),
  },
  {
    id: 'groq',
    name: 'Groq',
    envVar: 'GROQ_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'marathi', 'tamil',
    ]),
  },
  {
    id: 'cartesia',
    name: 'Cartesia',
    envVar: 'CARTESIA_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi',
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
    id: 'sarvam',
    name: 'Sarvam AI',
    envVar: 'SARVAM_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'odia', 'punjabi', 'tamil', 'telugu', 'gujarati',
    ]),
  },
  {
    id: 'elevenlabs',
    name: 'ElevenLabs',
    envVar: 'ELEVENLABS_API_KEY',
    languages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'odia', 'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi',
    ]),
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

export function getProvidersForLanguage(language: string, mode: 'tts' | 'stt'): ProviderInfo[] {
  return mode === 'tts' ? getTtsProvidersForLanguage(language) : getSttProvidersForLanguage(language);
}

// Keep backwards-compatible aliases
export { getTtsProvidersForLanguage as getProvidersForLanguageTts };
