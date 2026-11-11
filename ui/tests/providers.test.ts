import { describe, it, expect } from "vitest";
import {
  getSttProvidersForLanguage,
  getSttProviderById,
  outputDirName,
} from "../source/providers.js";

const ids = (language: string) => getSttProvidersForLanguage(language).map((p) => p.id);

describe("realtime STT providers", () => {
  it("offers every realtime provider for English and Hindi", () => {
    for (const language of ["english", "hindi"]) {
      expect(ids(language)).toEqual(
        expect.arrayContaining([
          "deepgram-flux", "openai-gpt-live-transcribe", "openai-gpt-transcribe",
          "sarvam-saaras-v4-realtime", "sarvam-saaras-v3-realtime", "sarvam-saaras-v4",
          "cartesia-ink-preview",
        ]),
      );
    }
  });

  it("limits Flux and Ink Preview to English and Hindi", () => {
    expect(ids("kannada")).toContain("openai-gpt-live-transcribe");
    expect(ids("kannada")).not.toContain("deepgram-flux");
    expect(ids("kannada")).not.toContain("cartesia-ink-preview");
    expect(ids("odia")).not.toContain("openai-gpt-live-transcribe");
    expect(ids("odia")).toEqual(
      expect.arrayContaining([
        "openai-gpt-transcribe", "sarvam-saaras-v4-realtime", "sarvam-saaras-v3-realtime",
      ]),
    );
  });

  it("uses each provider's API key", () => {
    expect(getSttProviderById("deepgram-flux")?.envVar).toBe("DEEPGRAM_API_KEY");
    expect(getSttProviderById("openai-gpt-live-transcribe")?.envVar).toBe("OPENAI_API_KEY");
    expect(getSttProviderById("openai-gpt-transcribe")?.envVar).toBe("OPENAI_API_KEY");
    expect(getSttProviderById("sarvam-saaras-v4-realtime")?.envVar).toBe("SARVAM_API_KEY");
    expect(getSttProviderById("sarvam-saaras-v3-realtime")?.envVar).toBe("SARVAM_API_KEY");
    expect(getSttProviderById("sarvam-saaras-v4")?.envVar).toBe("SARVAM_API_KEY");
    expect(getSttProviderById("cartesia-ink-preview")?.envVar).toBe("CARTESIA_API_KEY");
  });

  it("offers ElevenLabs Scribe v3 Live with the ElevenLabs key", () => {
    expect(ids("hindi")).toContain("elevenlabs-scribe-v3-live");
    expect(getSttProviderById("elevenlabs-scribe-v3-live")?.envVar).toBe("ELEVENLABS_API_KEY");
  });

  it("offers Gladia and Bodhan for the languages they list", () => {
    expect(ids("sindhi")).toEqual(
      expect.arrayContaining(["gladia-solaria-1", "bodhan-indic-transcribe"]),
    );
    expect(ids("odia")).toContain("bodhan-indic-transcribe");
    expect(ids("odia")).not.toContain("gladia-solaria-1");
    expect(getSttProviderById("gladia-solaria-1")?.envVar).toBe("GLADIA_API_KEY");
    expect(getSttProviderById("bodhan-indic-transcribe")?.envVar).toBe("BODHAN_API_KEY");
  });

  it("stores deepgram-flux results under the model it runs for the language", () => {
    expect(outputDirName("deepgram-flux", "stt", "english")).toBe("deepgram-flux-general-en");
    expect(outputDirName("deepgram-flux", "stt", "hindi")).toBe("deepgram-flux-general-multi");
    expect(outputDirName("deepgram-nova-3", "stt", "hindi")).toBe("deepgram-nova-3");
    expect(outputDirName("cartesia", "tts", "hindi")).toBe("cartesia");
  });
});
