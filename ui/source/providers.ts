export interface ProviderInfo {
  id: string;
  name: string;
  envVar: string;
  isFilePath?: boolean;
  ttsLanguages: Set<string>;
}

export const TTS_PROVIDERS: ProviderInfo[] = [
  {
    id: 'openai',
    name: 'OpenAI',
    envVar: 'OPENAI_API_KEY',
    ttsLanguages: new Set([
      'english', 'hindi', 'kannada', 'marathi', 'tamil',
    ]),
  },
  {
    id: 'google',
    name: 'Google Cloud',
    envVar: 'GOOGLE_APPLICATION_CREDENTIALS',
    isFilePath: true,
    ttsLanguages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'odia', 'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi',
    ]),
  },
  {
    id: 'elevenlabs',
    name: 'ElevenLabs',
    envVar: 'ELEVENLABS_API_KEY',
    ttsLanguages: new Set([
      'english', 'hindi', 'sindhi', 'tamil',
    ]),
  },
  {
    id: 'cartesia',
    name: 'Cartesia',
    envVar: 'CARTESIA_API_KEY',
    ttsLanguages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'punjabi', 'tamil', 'telugu', 'gujarati',
    ]),
  },
  {
    id: 'groq',
    name: 'Groq',
    envVar: 'GROQ_API_KEY',
    ttsLanguages: new Set(['english']),
  },
  {
    id: 'sarvam',
    name: 'Sarvam AI',
    envVar: 'SARVAM_API_KEY',
    ttsLanguages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'odia', 'punjabi', 'tamil', 'telugu', 'gujarati',
    ]),
  },
  {
    id: 'smallest',
    name: 'Smallest AI',
    envVar: 'SMALLEST_API_KEY',
    ttsLanguages: new Set([
      'english', 'hindi', 'kannada', 'bengali', 'malayalam',
      'marathi', 'odia', 'tamil', 'telugu', 'gujarati',
    ]),
  },
];

export const LANGUAGES = [
  'english', 'hindi', 'kannada', 'bengali', 'malayalam',
  'marathi', 'odia', 'punjabi', 'tamil', 'telugu', 'gujarati', 'sindhi',
];

export function getProviderById(id: string): ProviderInfo | undefined {
  return TTS_PROVIDERS.find(p => p.id === id);
}

export function getProvidersForLanguage(language: string): ProviderInfo[] {
  return TTS_PROVIDERS.filter(p => p.ttsLanguages.has(language));
}
