import { describe, it, expect } from "vitest";
import { getSttProvidersForLanguage, getSttProviderById } from "../source/providers.js";

const ids = (language: string) => getSttProvidersForLanguage(language).map((p) => p.id);

describe("realtime STT providers", () => {
  it("offers every realtime provider for English and Hindi", () => {
    for (const language of ["english", "hindi"]) {
      expect(ids(language)).toEqual(
        expect.arrayContaining([
          "deepgram-flux", "openai-live", "openai-transcribe", "sarvam-v4", "sarvam-v3-realtime",
          "sarvam-v4-manual", "cartesia-ink-preview",
        ]),
      );
    }
  });

  it("limits Flux and Ink Preview to English and Hindi", () => {
    expect(ids("kannada")).toContain("openai-live");
    expect(ids("kannada")).not.toContain("deepgram-flux");
    expect(ids("kannada")).not.toContain("cartesia-ink-preview");
    expect(ids("odia")).not.toContain("openai-live");
    expect(ids("odia")).toEqual(
      expect.arrayContaining(["openai-transcribe", "sarvam-v4", "sarvam-v3-realtime"]),
    );
  });

  it("uses each provider's API key", () => {
    expect(getSttProviderById("deepgram-flux")?.envVar).toBe("DEEPGRAM_API_KEY");
    expect(getSttProviderById("openai-live")?.envVar).toBe("OPENAI_API_KEY");
    expect(getSttProviderById("openai-transcribe")?.envVar).toBe("OPENAI_API_KEY");
    expect(getSttProviderById("sarvam-v4")?.envVar).toBe("SARVAM_API_KEY");
    expect(getSttProviderById("sarvam-v3-realtime")?.envVar).toBe("SARVAM_API_KEY");
    expect(getSttProviderById("sarvam-v4-manual")?.envVar).toBe("SARVAM_API_KEY");
    expect(getSttProviderById("cartesia-ink-preview")?.envVar).toBe("CARTESIA_API_KEY");
  });
});
