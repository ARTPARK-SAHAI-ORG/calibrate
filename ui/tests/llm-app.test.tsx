/**
 * Tests for LlmTestsApp — step navigation and cmdArgs assembly.
 *
 * Strategy:
 *  - vi.mock() for node:fs, node:child_process, ./shared.js, ./credentials.js
 *  - ink-testing-library render() to mount component
 *  - stdin.write() to send keypresses
 *  - act() + small async delays to flush React effects
 */

import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, cleanup } from "ink-testing-library";
import { EventEmitter } from "node:events";

// ─── Mocks ────────────────────────────────────────────────────────────────────

// Track spawn calls so tests can inspect args
const spawnMock = vi.fn();

vi.mock("node:fs", () => {
  const existsSyncMock = vi.fn(() => true);
  const readFileSyncMock = vi.fn((_path: string, _enc?: string) =>
    JSON.stringify({ agent_url: "http://localhost:8080" }),
  );
  const readdirSyncMock = vi.fn(() => []);
  return {
    default: {
      existsSync: existsSyncMock,
      readFileSync: readFileSyncMock,
      readdirSync: readdirSyncMock,
    },
    existsSync: existsSyncMock,
    readFileSync: readFileSyncMock,
    readdirSync: readdirSyncMock,
  };
});

vi.mock("node:child_process", () => {
  return {
    spawn: spawnMock,
  };
});

vi.mock("../source/shared.js", () => ({
  findCalibrateBin: vi.fn(() => ({ cmd: "calibrate-agent", args: [] })),
  stripAnsi: (s: string) => s,
}));

vi.mock("../source/credentials.js", () => ({
  getCredential: vi.fn(() => "sk-fake-key"),
  saveCredential: vi.fn(),
}));

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Create a minimal fake ChildProcess with stdout/stderr EventEmitters */
function makeFakeProc() {
  const proc = new EventEmitter() as NodeJS.EventEmitter & {
    stdout: EventEmitter;
    stderr: EventEmitter;
    kill: () => void;
  };
  proc.stdout = new EventEmitter();
  proc.stderr = new EventEmitter();
  proc.kill = vi.fn();
  return proc;
}

/** Wait for React effects to flush */
function wait(ms = 50): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Type characters one at a time then press Enter */
async function typeAndSubmit(
  stdin: { write: (s: string) => void },
  text: string,
) {
  for (const ch of text) {
    stdin.write(ch);
    await wait(10);
  }
  // Enter key
  stdin.write("\r");
  await wait(50);
}

/** Press Enter (select current item in SelectInput or submit TextInput) */
async function pressEnter(stdin: { write: (s: string) => void }) {
  stdin.write("\r");
  await wait(50);
}

/** Press down arrow */
async function pressDown(stdin: { write: (s: string) => void }) {
  stdin.write("\u001B[B");
  await wait(30);
}

// ─── Import component (after mocks are set up) ────────────────────────────────

// Dynamic import to ensure mocks are in place before module loads
let LlmTestsApp: React.ComponentType<{ onBack?: () => void }>;

// ─── Tests ────────────────────────────────────────────────────────────────────

describe("LlmTestsApp", () => {
  beforeEach(async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    spawnMock.mockReset();

    // Default spawn: returns a fake process that does nothing
    spawnMock.mockImplementation(() => makeFakeProc());

    // Dynamic import so mocks are applied
    const mod = await import("../source/llm-app.js");
    LlmTestsApp = mod.LlmTestsApp;
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.resetModules();
  });

  // ─── Helper to advance past init step ──────────────────────────────────────
  async function renderAndInit() {
    const { stdout, stdin } = render(<LlmTestsApp />);
    // Wait for init useEffect to run (transitions init → config-path)
    await wait(100);
    return { stdout, stdin };
  }

  // ──────────────────────────────────────────────────────────────────────────
  // config-path step tests
  // ──────────────────────────────────────────────────────────────────────────

  describe("config-path step", () => {
    it("shows config-path prompt after init", async () => {
      const { stdout } = await renderAndInit();
      expect(stdout.lastFrame()).toContain("Config file");
    });

    it("navigates to agent-mode when config has agent_url", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValueOnce(
        JSON.stringify({ agent_url: "http://localhost:8080" }),
      );

      const { stdout, stdin } = await renderAndInit();

      // Submit a config path
      await typeAndSubmit(stdin, "./config.json");

      expect(stdout.lastFrame()).toContain("How do you want to run tests");
    });

    it("navigates to provider step when config has no agent_url", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValueOnce(
        JSON.stringify({}),
      );

      const { stdout, stdin } = await renderAndInit();

      await typeAndSubmit(stdin, "./config.json");

      expect(stdout.lastFrame()).toContain("Provider:");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // agent-mode step tests
  // ──────────────────────────────────────────────────────────────────────────

  describe("agent-mode step", () => {
    async function reachAgentMode() {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://localhost:8080" }),
      );

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json");
      // Should now be on agent-mode
      return { stdout, stdin };
    }

    it("shows agent-mode options", async () => {
      const { stdout } = await reachAgentMode();
      expect(stdout.lastFrame()).toContain("How do you want to run tests");
    });

    it("selecting Single test navigates to agent-verify", async () => {
      const { stdout, stdin } = await reachAgentMode();

      // First item is "Single test" — just press Enter to select it
      await pressEnter(stdin);

      expect(stdout.lastFrame()).toContain("Verifying agent connection");
    });

    it("selecting Benchmark navigates to agent-model-entry", async () => {
      const { stdout, stdin } = await reachAgentMode();

      // Navigate down to "Benchmark" then press Enter
      await pressDown(stdin);
      await pressEnter(stdin);

      expect(stdout.lastFrame()).toContain("model");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // agent-model-entry step
  // ──────────────────────────────────────────────────────────────────────────

  describe("agent-model-entry step", () => {
    async function reachModelEntry() {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://localhost:8080" }),
      );

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressDown(stdin); // select Benchmark
      await pressEnter(stdin); // → agent-model-entry
      return { stdout, stdin };
    }

    it("type model name + Enter navigates to agent-verify", async () => {
      const { stdout, stdin } = await reachModelEntry();

      await typeAndSubmit(stdin, "gpt-4.1");

      expect(stdout.lastFrame()).toContain("Verifying agent connection");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // agent-verify step — spawn close code 0 (single run)
  // ──────────────────────────────────────────────────────────────────────────

  describe("agent-verify step — single run", () => {
    it("code 0 navigates to output-dir after 1s delay", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://localhost:8080" }),
      );

      let capturedProc: ReturnType<typeof makeFakeProc> | null = null;
      spawnMock.mockImplementation(() => {
        capturedProc = makeFakeProc();
        return capturedProc;
      });

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressEnter(stdin); // select Single test → agent-verify

      // Verify step useEffect has spawned
      await wait(50);
      expect(capturedProc).not.toBeNull();

      // Simulate successful close
      capturedProc!.emit("close", 0);
      await wait(50);
      expect(stdout.lastFrame()).toContain("Connection verified");

      // Advance fake timers past the 1000ms delay
      vi.advanceTimersByTime(1100);
      await wait(50);

      expect(stdout.lastFrame()).toContain("Output directory");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // agent-verify step — spawn close code 0 (benchmark run)
  // ──────────────────────────────────────────────────────────────────────────

  describe("agent-verify step — benchmark run", () => {
    it("code 0 navigates to agent-model-confirm after 1s delay", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://localhost:8080" }),
      );

      let capturedProc: ReturnType<typeof makeFakeProc> | null = null;
      spawnMock.mockImplementation(() => {
        capturedProc = makeFakeProc();
        return capturedProc;
      });

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressDown(stdin); // select Benchmark
      await pressEnter(stdin); // → agent-model-entry
      await typeAndSubmit(stdin, "gpt-4.1"); // → agent-verify

      await wait(50);
      expect(capturedProc).not.toBeNull();

      capturedProc!.emit("close", 0);
      await wait(50);
      expect(stdout.lastFrame()).toContain("Connection verified");

      vi.advanceTimersByTime(1100);
      await wait(50);

      expect(stdout.lastFrame()).toContain("Add another model");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // agent-verify — non-zero exit code (stays on verify with error)
  // ──────────────────────────────────────────────────────────────────────────

  describe("agent-verify step — failure", () => {
    it("non-zero exit stays on verify and shows error", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://localhost:8080" }),
      );

      let capturedProc: ReturnType<typeof makeFakeProc> | null = null;
      spawnMock.mockImplementation(() => {
        capturedProc = makeFakeProc();
        return capturedProc;
      });

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json");
      await pressEnter(stdin); // single test → agent-verify

      await wait(50);
      capturedProc!.stdout.emit("data", Buffer.from("Connection failed\n"));
      capturedProc!.emit("close", 1);
      await wait(50);

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Verification failed");
      expect(frame).toContain("Connection failed");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // agent-model-confirm step
  // ──────────────────────────────────────────────────────────────────────────

  describe("agent-model-confirm step", () => {
    async function reachModelConfirm() {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://localhost:8080" }),
      );

      let capturedProc: ReturnType<typeof makeFakeProc> | null = null;
      spawnMock.mockImplementation(() => {
        capturedProc = makeFakeProc();
        return capturedProc;
      });

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressDown(stdin); // select Benchmark
      await pressEnter(stdin); // → agent-model-entry
      await typeAndSubmit(stdin, "gpt-4.1"); // → agent-verify

      await wait(50);
      capturedProc!.emit("close", 0);
      await wait(50);
      vi.advanceTimersByTime(1100);
      await wait(50);
      // Now on agent-model-confirm

      return { stdout, stdin };
    }

    it("selecting 'add another' navigates to agent-model-entry", async () => {
      const { stdout, stdin } = await reachModelConfirm();

      // First item is "Yes, add another model"
      await pressEnter(stdin);

      const frame = stdout.lastFrame() ?? "";
      // agent-model-entry shows model entry prompt
      expect(frame).toContain("model");
    });

    it("selecting 'continue' navigates to output-dir", async () => {
      const { stdout, stdin } = await reachModelConfirm();

      // Navigate down to "No, continue" then press Enter
      await pressDown(stdin);
      await pressEnter(stdin);

      expect(stdout.lastFrame()).toContain("Output directory");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // cmdArgs tests — what spawn is called with in startModel
  // ──────────────────────────────────────────────────────────────────────────

  describe("startModel cmdArgs", () => {
    /**
     * Navigate all the way to the running step for a single agent run,
     * then return the args spawn was called with.
     */
    async function getSingleAgentSpawnArgs(): Promise<string[]> {
      const fs = await import("node:fs");
      // existsSync: true for config file, false for output dirs
      (fs.default.existsSync as ReturnType<typeof vi.fn>).mockImplementation(
        (_p: string) => true,
      );
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockImplementation(
        (p: string) => {
          if (typeof p === "string" && p.endsWith(".json")) {
            return JSON.stringify({ agent_url: "http://localhost:8080" });
          }
          return "";
        },
      );
      (fs.default.readdirSync as ReturnType<typeof vi.fn>).mockReturnValue([]);

      const verifyProc = makeFakeProc();
      const runProc = makeFakeProc();
      let spawnCallCount = 0;

      spawnMock.mockImplementation(() => {
        spawnCallCount++;
        if (spawnCallCount === 1) return verifyProc; // verify spawn
        return runProc; // run spawn
      });

      const { stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressEnter(stdin); // single → agent-verify

      await wait(50);
      verifyProc.emit("close", 0);
      await wait(50);
      vi.advanceTimersByTime(1100);
      await wait(100); // → output-dir

      // Submit the output dir
      stdin.write("\r");
      await wait(100); // → running (no existing dirs, api key present)

      // startModel is called from useEffect
      await wait(100);

      // Find the run spawn call (second call)
      const calls = spawnMock.mock.calls;
      const runCall = calls.find(
        (c) => Array.isArray(c[1]) && c[1].includes("--skip-verify"),
      );
      return runCall ? (runCall[1] as string[]) : [];
    }

    it("single agent run: args contain --skip-verify, no -m", async () => {
      const args = await getSingleAgentSpawnArgs();
      expect(args).toContain("--skip-verify");
      expect(args).not.toContain("-m");
    });

    /**
     * Navigate to benchmark run with model "gpt-4.1".
     */
    async function getBenchmarkAgentSpawnArgs(): Promise<string[]> {
      const fs = await import("node:fs");
      (fs.default.existsSync as ReturnType<typeof vi.fn>).mockImplementation(
        () => true,
      );
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockImplementation(
        (p: string) => {
          if (typeof p === "string" && p.endsWith(".json")) {
            return JSON.stringify({ agent_url: "http://localhost:8080" });
          }
          return "";
        },
      );
      (fs.default.readdirSync as ReturnType<typeof vi.fn>).mockReturnValue([]);

      const verifyProc = makeFakeProc();
      const runProc = makeFakeProc();
      let spawnCallCount = 0;

      spawnMock.mockImplementation(() => {
        spawnCallCount++;
        if (spawnCallCount === 1) return verifyProc;
        return runProc;
      });

      const { stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressDown(stdin); // select Benchmark
      await pressEnter(stdin); // → agent-model-entry
      await typeAndSubmit(stdin, "gpt-4.1"); // → agent-verify

      await wait(50);
      verifyProc.emit("close", 0);
      await wait(50);
      vi.advanceTimersByTime(1100);
      await wait(100); // → agent-model-confirm

      // Select "continue"
      await pressDown(stdin);
      await pressEnter(stdin);
      await wait(100); // → output-dir

      stdin.write("\r");
      await wait(100); // → running

      await wait(100);

      const calls = spawnMock.mock.calls;
      const runCall = calls.find(
        (c) =>
          Array.isArray(c[1]) &&
          c[1].includes("--skip-verify") &&
          c[1].includes("-m"),
      );
      return runCall ? (runCall[1] as string[]) : [];
    }

    it("benchmark agent run (gpt-4.1): args contain --skip-verify, -m, gpt-4.1", async () => {
      const args = await getBenchmarkAgentSpawnArgs();
      expect(args).toContain("--skip-verify");
      expect(args).toContain("-m");
      expect(args).toContain("gpt-4.1");
    });

    /**
     * Navigate to internal model run (gpt-4.1, openrouter).
     */
    async function getInternalModelSpawnArgs(): Promise<string[]> {
      const fs = await import("node:fs");
      (fs.default.existsSync as ReturnType<typeof vi.fn>).mockImplementation(
        () => true,
      );
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockImplementation(
        (p: string) => {
          if (typeof p === "string" && p.endsWith(".json")) {
            // No agent_url → internal model path
            return JSON.stringify({});
          }
          return "";
        },
      );
      (fs.default.readdirSync as ReturnType<typeof vi.fn>).mockReturnValue([]);

      const runProc = makeFakeProc();
      spawnMock.mockImplementation(() => runProc);

      const { stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → provider
      // Accept default OpenRouter (first item)
      await pressEnter(stdin); // → enter-model
      await typeAndSubmit(stdin, "gpt-4.1"); // → model-confirm
      // Continue with selected models (press down to select "No, continue")
      await pressDown(stdin);
      await pressEnter(stdin);
      await wait(100); // → output-dir

      stdin.write("\r");
      await wait(100); // → api-keys or running

      // api-keys: submit fake key for OPENAI_API_KEY
      // (getCredential returns "sk-fake-key" so it might skip api-keys)
      await wait(100); // → running

      await wait(100);

      const calls = spawnMock.mock.calls;
      // Find the call that has -m and -p (internal model run)
      const runCall = calls.find(
        (c) =>
          Array.isArray(c[1]) && c[1].includes("-m") && c[1].includes("-p"),
      );
      return runCall ? (runCall[1] as string[]) : [];
    }

    it("internal model run (gpt-4.1, openrouter): args contain -m openai/gpt-4.1 -p openrouter, no --skip-verify", async () => {
      const args = await getInternalModelSpawnArgs();
      expect(args).toContain("-m");
      expect(args).toContain("gpt-4.1");
      expect(args).toContain("-p");
      expect(args).toContain("openrouter");
      expect(args).not.toContain("--skip-verify");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // internal model flow — provider step
  // ──────────────────────────────────────────────────────────────────────────

  describe("internal model flow — provider step", () => {
    async function reachProvider() {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({}), // no agent_url → internal path
      );
      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → provider
      return { stdout, stdin };
    }

    it("shows provider options after internal config", async () => {
      const { stdout } = await reachProvider();
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("OpenRouter");
      expect(frame).toContain("OpenAI");
    });

    it("selecting OpenRouter navigates to enter-model", async () => {
      const { stdout, stdin } = await reachProvider();
      // First item is OpenRouter — press Enter
      await pressEnter(stdin);
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toMatch(/Model:|Enter model|openrouter/i);
    });

    it("selecting OpenAI navigates to enter-model", async () => {
      const { stdout, stdin } = await reachProvider();
      await pressDown(stdin); // select OpenAI
      await pressEnter(stdin);
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toMatch(/Model:|Enter model|openai/i);
    });

    it("provider step shows correct model examples for OpenRouter", async () => {
      const { stdout, stdin } = await reachProvider();
      await pressEnter(stdin); // select OpenRouter → enter-model
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("openai/gpt-4.1");
    });

    it("provider step shows correct model examples for OpenAI", async () => {
      const { stdout, stdin } = await reachProvider();
      await pressDown(stdin); // select OpenAI
      await pressEnter(stdin); // → enter-model
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("gpt-4.1");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // internal model flow — enter-model step
  // ──────────────────────────────────────────────────────────────────────────

  describe("internal model flow — enter-model step", () => {
    async function reachEnterModel(
      provider: "openrouter" | "openai" = "openrouter",
    ) {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({}),
      );
      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → provider
      if (provider === "openai") {
        await pressDown(stdin); // select OpenAI
      }
      await pressEnter(stdin); // → enter-model
      return { stdout, stdin };
    }

    it("typing model and Enter navigates to model-confirm", async () => {
      const { stdout, stdin } = await reachEnterModel();
      await typeAndSubmit(stdin, "gpt-4.1");
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("gpt-4.1");
      // model-confirm shows "Add another model?"
      expect(frame).toContain("Add another model");
    });

    it("empty Enter uses default model for openrouter", async () => {
      const { stdout, stdin } = await reachEnterModel("openrouter");
      // Just press Enter without typing → uses default openrouter model
      await pressEnter(stdin);
      const frame = stdout.lastFrame() ?? "";
      // model-confirm should show the default model
      expect(frame).toContain("openai/gpt-4.1");
    });

    it("duplicate model shows error", async () => {
      const { stdout, stdin } = await reachEnterModel();
      // Add model first time
      await typeAndSubmit(stdin, "gpt-4.1"); // → model-confirm
      // Go back to enter-model: first item (index 0) is "Yes, add another model"
      await pressEnter(stdin); // → enter-model
      // Type the same model again
      await typeAndSubmit(stdin, "gpt-4.1");
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("already selected");
    });

    it("model-confirm shows added model", async () => {
      const { stdout, stdin } = await reachEnterModel();
      await typeAndSubmit(stdin, "my-custom-model");
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("my-custom-model");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // internal model flow — model-confirm step
  // ──────────────────────────────────────────────────────────────────────────

  describe("internal model flow — model-confirm step", () => {
    async function reachModelConfirmInternal() {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({}),
      );
      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → provider
      await pressEnter(stdin); // select OpenRouter → enter-model
      await typeAndSubmit(stdin, "openai/gpt-4.1"); // → model-confirm
      return { stdout, stdin };
    }

    it("selecting add another returns to enter-model", async () => {
      const { stdout, stdin } = await reachModelConfirmInternal();
      // First item is "Yes, add another model"
      await pressEnter(stdin);
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toMatch(/Model:|Add another model:/i);
    });

    it("selecting continue navigates to output-dir", async () => {
      const { stdout, stdin } = await reachModelConfirmInternal();
      // Second item is "No, continue"
      await pressDown(stdin);
      await pressEnter(stdin);
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Output directory");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // shared steps — output-dir step
  // ──────────────────────────────────────────────────────────────────────────

  describe("shared steps — output-dir step", () => {
    /** Fastest path to output-dir: agent-single */
    async function reachOutputDirViaSingle() {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://fake-agent/chat" }),
      );
      (fs.default.readdirSync as ReturnType<typeof vi.fn>).mockReturnValue([]);

      let capturedProc: ReturnType<typeof makeFakeProc> | null = null;
      spawnMock.mockImplementation(() => {
        capturedProc = makeFakeProc();
        return capturedProc;
      });

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressEnter(stdin); // select Single → agent-verify

      await wait(50);
      capturedProc!.emit("close", 0);
      await wait(50);
      vi.advanceTimersByTime(1100);
      await wait(100); // → output-dir

      return { stdout, stdin };
    }

    it("shows output directory prompt", async () => {
      const { stdout } = await reachOutputDirViaSingle();
      expect(stdout.lastFrame()).toContain("Output directory");
    });

    it("submitting goes to running when api keys present", async () => {
      const credentials = await import("../source/credentials.js");
      (credentials.getCredential as ReturnType<typeof vi.fn>).mockReturnValue(
        "sk-key",
      );

      const { stdout, stdin } = await reachOutputDirViaSingle();
      await pressEnter(stdin); // submit default dir
      await wait(100);

      const frame = stdout.lastFrame() ?? "";
      // Should be in "running" or show Config:
      expect(frame).not.toContain("Output directory:");
    });

    it("submitting goes to api-keys when keys missing", async () => {
      const credentials = await import("../source/credentials.js");
      (credentials.getCredential as ReturnType<typeof vi.fn>).mockReturnValue(
        null,
      );

      const { stdout, stdin } = await reachOutputDirViaSingle();
      await pressEnter(stdin);
      await wait(100);

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("OPENAI_API_KEY");
    });

    it("submitting with existing dirs shows output-dir-confirm", async () => {
      const fs = await import("node:fs");
      const credentials = await import("../source/credentials.js");
      // Ensure credentials are present so we don't go to api-keys
      (credentials.getCredential as ReturnType<typeof vi.fn>).mockReturnValue(
        "sk-key",
      );

      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://fake-agent/chat" }),
      );
      (fs.default.existsSync as ReturnType<typeof vi.fn>).mockReturnValue(true);
      // Make readdirSync return entries indicating existing data
      (fs.default.readdirSync as ReturnType<typeof vi.fn>).mockImplementation(
        (p: unknown, _opts?: unknown) => {
          // For the inner dir (content check): return something non-empty
          return ["results.json"];
        },
      );

      let capturedProc: ReturnType<typeof makeFakeProc> | null = null;
      spawnMock.mockImplementation(() => {
        capturedProc = makeFakeProc();
        return capturedProc;
      });

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressEnter(stdin); // single → agent-verify

      await wait(50);
      capturedProc!.emit("close", 0);
      await wait(50);
      vi.advanceTimersByTime(1100);
      await wait(100); // → output-dir

      // Now override readdirSync to return a directory entry for the outer dir
      (fs.default.readdirSync as ReturnType<typeof vi.fn>).mockImplementation(
        (p: unknown, _opts?: unknown) => {
          const pStr = String(p);
          if (pStr.endsWith("out") || pStr === "./out") {
            return [{ name: "model1", isDirectory: () => true }];
          }
          return ["results.json"];
        },
      );

      await pressEnter(stdin); // submit default dir "./out"
      await wait(100);

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Existing data found");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // shared steps — api-keys step
  // ──────────────────────────────────────────────────────────────────────────

  describe("shared steps — api-keys step", () => {
    async function reachApiKeys() {
      const credentials = await import("../source/credentials.js");
      (credentials.getCredential as ReturnType<typeof vi.fn>).mockReturnValue(
        null,
      );

      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://fake-agent/chat" }),
      );
      (fs.default.readdirSync as ReturnType<typeof vi.fn>).mockReturnValue([]);

      let capturedProc: ReturnType<typeof makeFakeProc> | null = null;
      spawnMock.mockImplementation(() => {
        capturedProc = makeFakeProc();
        return capturedProc;
      });

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressEnter(stdin); // single → agent-verify

      await wait(50);
      capturedProc!.emit("close", 0);
      await wait(50);
      vi.advanceTimersByTime(1100);
      await wait(100); // → output-dir

      await pressEnter(stdin); // submit output dir → api-keys
      await wait(100);

      return { stdout, stdin };
    }

    it("shows key name prompt", async () => {
      const { stdout } = await reachApiKeys();
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("OPENAI_API_KEY");
    });

    it("entering key advances to running", async () => {
      const { stdout, stdin } = await reachApiKeys();
      await typeAndSubmit(stdin, "sk-test-key");
      await wait(100);
      const frame = stdout.lastFrame() ?? "";
      // Should have left api-keys step
      expect(frame).not.toContain("OPENAI_API_KEY:");
    });

    it("entering first of two keys advances to second key", async () => {
      const credentials = await import("../source/credentials.js");
      // Both keys missing — first call returns null, second also null
      (credentials.getCredential as ReturnType<typeof vi.fn>).mockReturnValue(
        null,
      );

      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({}), // internal path → needs both OPENAI and OPENROUTER keys
      );
      (fs.default.readdirSync as ReturnType<typeof vi.fn>).mockReturnValue([]);

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → provider
      await pressEnter(stdin); // OpenRouter → enter-model
      await typeAndSubmit(stdin, "openai/gpt-4.1"); // → model-confirm
      await pressDown(stdin);
      await pressEnter(stdin); // continue → output-dir
      await wait(100);

      await pressEnter(stdin); // submit output dir → api-keys (both keys missing)
      await wait(100);

      // Enter first key
      await typeAndSubmit(stdin, "sk-openai-key");
      await wait(100);

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("OPENROUTER_API_KEY");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // back navigation
  // ──────────────────────────────────────────────────────────────────────────

  describe("back navigation", () => {
    async function pressEsc(stdin: { write: (s: string) => void }) {
      stdin.write("\u001B");
      await wait(50);
    }

    it("Esc from provider goes to config-path", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({}),
      );
      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → provider
      await pressEsc(stdin); // → config-path
      expect(stdout.lastFrame()).toContain("Config file:");
    });

    it("Esc from enter-model goes to provider", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({}),
      );
      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → provider
      await pressEnter(stdin); // → enter-model
      await pressEsc(stdin); // → provider
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Provider:");
    });

    it("Esc from model-confirm goes to enter-model", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({}),
      );
      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → provider
      await pressEnter(stdin); // → enter-model
      await typeAndSubmit(stdin, "gpt-4.1"); // → model-confirm
      await pressEsc(stdin); // → enter-model
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toMatch(/Model:|Add another model:/i);
    });

    it("Esc from agent-mode goes to config-path", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://fake-agent/chat" }),
      );
      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressEsc(stdin); // → config-path
      expect(stdout.lastFrame()).toContain("Config file:");
    });

    it("Esc from agent-model-entry with no models goes to agent-mode", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://fake-agent/chat" }),
      );
      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressDown(stdin); // select Benchmark
      await pressEnter(stdin); // → agent-model-entry
      await pressEsc(stdin); // → agent-mode (no models yet)
      expect(stdout.lastFrame()).toContain("How do you want to run tests");
    });

    it("Esc from agent-model-entry with models goes to agent-model-confirm", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://fake-agent/chat" }),
      );

      let capturedProc: ReturnType<typeof makeFakeProc> | null = null;
      spawnMock.mockImplementation(() => {
        capturedProc = makeFakeProc();
        return capturedProc;
      });

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressDown(stdin); // select Benchmark
      await pressEnter(stdin); // → agent-model-entry
      await typeAndSubmit(stdin, "gpt-4.1"); // → agent-verify

      await wait(50);
      capturedProc!.emit("close", 0);
      await wait(50);
      vi.advanceTimersByTime(1100);
      await wait(100); // → agent-model-confirm

      // Now go to agent-model-entry to add another
      await pressEnter(stdin); // "Yes, add another model" → agent-model-entry
      await pressEsc(stdin); // → agent-model-confirm (has 1 model)

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Add another model");
    });

    it("Esc from agent-verify single goes to agent-mode", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://fake-agent/chat" }),
      );
      spawnMock.mockImplementation(() => makeFakeProc());

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressEnter(stdin); // single → agent-verify
      await wait(50);
      await pressEsc(stdin); // → agent-mode
      expect(stdout.lastFrame()).toContain("How do you want to run tests");
    });

    it("Esc from agent-verify benchmark removes last model and goes to agent-model-entry", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://fake-agent/chat" }),
      );
      spawnMock.mockImplementation(() => makeFakeProc());

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressDown(stdin); // Benchmark
      await pressEnter(stdin); // → agent-model-entry
      await typeAndSubmit(stdin, "gpt-4.1"); // → agent-verify (adds gpt-4.1)
      await wait(50);
      await pressEsc(stdin); // → agent-model-entry, gpt-4.1 removed
      await wait(50);

      const frame = stdout.lastFrame() ?? "";
      // Should be on agent-model-entry (not agent-model-confirm)
      expect(frame).toMatch(/Model:|gemma-4-26b/i);
      expect(frame).not.toContain("gpt-4.1");
    });

    it("Esc from output-dir agent single goes to agent-verify", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://fake-agent/chat" }),
      );
      (fs.default.readdirSync as ReturnType<typeof vi.fn>).mockReturnValue([]);

      let capturedProc: ReturnType<typeof makeFakeProc> | null = null;
      spawnMock.mockImplementation(() => {
        capturedProc = makeFakeProc();
        return capturedProc;
      });

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressEnter(stdin); // single → agent-verify

      await wait(50);
      capturedProc!.emit("close", 0);
      await wait(50);
      vi.advanceTimersByTime(1100);
      await wait(100); // → output-dir

      await pressEsc(stdin); // → agent-verify
      await wait(50);
      expect(stdout.lastFrame()).toContain("Verifying agent connection");
    });

    it("Esc from output-dir agent benchmark goes to agent-model-confirm", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://fake-agent/chat" }),
      );
      (fs.default.readdirSync as ReturnType<typeof vi.fn>).mockReturnValue([]);

      let capturedProc: ReturnType<typeof makeFakeProc> | null = null;
      spawnMock.mockImplementation(() => {
        capturedProc = makeFakeProc();
        return capturedProc;
      });

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressDown(stdin); // Benchmark
      await pressEnter(stdin); // → agent-model-entry
      await typeAndSubmit(stdin, "gpt-4.1"); // → agent-verify

      await wait(50);
      capturedProc!.emit("close", 0);
      await wait(50);
      vi.advanceTimersByTime(1100);
      await wait(100); // → agent-model-confirm

      await pressDown(stdin); // select "continue"
      await pressEnter(stdin); // → output-dir
      await wait(100);

      await pressEsc(stdin); // → agent-model-confirm
      await wait(50);
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Add another model");
    });

    it("Esc from output-dir internal goes to model-confirm", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({}),
      );
      (fs.default.readdirSync as ReturnType<typeof vi.fn>).mockReturnValue([]);

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → provider
      await pressEnter(stdin); // OpenRouter → enter-model
      await typeAndSubmit(stdin, "openai/gpt-4.1"); // → model-confirm
      await pressDown(stdin); // select "continue"
      await pressEnter(stdin); // → output-dir
      await wait(100);

      await pressEsc(stdin); // → model-confirm
      await wait(50);
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Add another model");
    });

    it("Esc from output-dir-confirm goes to output-dir", async () => {
      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://fake-agent/chat" }),
      );
      // Make readdirSync return existing dirs for output-dir-confirm trigger
      (fs.default.readdirSync as ReturnType<typeof vi.fn>).mockImplementation(
        (p: unknown, _opts?: unknown) => {
          if (typeof p === "string" && p === "./out") {
            return [{ name: "model1", isDirectory: () => true }];
          }
          return ["results.json"];
        },
      );
      (fs.default.existsSync as ReturnType<typeof vi.fn>).mockReturnValue(true);

      let capturedProc: ReturnType<typeof makeFakeProc> | null = null;
      spawnMock.mockImplementation(() => {
        capturedProc = makeFakeProc();
        return capturedProc;
      });

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressEnter(stdin); // single → agent-verify

      await wait(50);
      capturedProc!.emit("close", 0);
      await wait(50);
      vi.advanceTimersByTime(1100);
      await wait(100); // → output-dir

      await pressEnter(stdin); // submit default dir → output-dir-confirm
      await wait(100);
      expect(stdout.lastFrame()).toContain("Existing data found");

      await pressEsc(stdin); // → output-dir
      await wait(50);
      expect(stdout.lastFrame()).toContain("Output directory");
    });

    it("Esc from api-keys goes to output-dir", async () => {
      const credentials = await import("../source/credentials.js");
      (credentials.getCredential as ReturnType<typeof vi.fn>).mockReturnValue(
        null,
      );

      const fs = await import("node:fs");
      (fs.default.readFileSync as ReturnType<typeof vi.fn>).mockReturnValue(
        JSON.stringify({ agent_url: "http://fake-agent/chat" }),
      );
      (fs.default.readdirSync as ReturnType<typeof vi.fn>).mockReturnValue([]);

      let capturedProc: ReturnType<typeof makeFakeProc> | null = null;
      spawnMock.mockImplementation(() => {
        capturedProc = makeFakeProc();
        return capturedProc;
      });

      const { stdout, stdin } = await renderAndInit();
      await typeAndSubmit(stdin, "./config.json"); // → agent-mode
      await pressEnter(stdin); // single → agent-verify

      await wait(50);
      capturedProc!.emit("close", 0);
      await wait(50);
      vi.advanceTimersByTime(1100);
      await wait(100); // → output-dir

      await pressEnter(stdin); // → api-keys
      await wait(100);
      expect(stdout.lastFrame()).toContain("OPENAI_API_KEY");

      await pressEsc(stdin); // → output-dir
      await wait(50);
      expect(stdout.lastFrame()).toContain("Output directory");
    });
  });
});
