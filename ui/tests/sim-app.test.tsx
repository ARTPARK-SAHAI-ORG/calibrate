/**
 * Tests for SimulationsApp — step navigation and cmdArgs assembly.
 *
 * Strategy:
 *  - vi.mock() for node:fs, node:child_process, ./shared.js, ./credentials.js
 *  - ink-testing-library render() to mount component
 *  - stdin.write() to send keypresses
 *  - small async delays to flush React effects
 */

import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, cleanup } from "ink-testing-library";
import { EventEmitter } from "node:events";

// ─── Mocks ────────────────────────────────────────────────────────────────────

// Track spawn calls so tests can inspect args
const spawnMock = vi.fn();

// Default fs mock: calibrate agent config (no agent_url)
vi.mock("node:fs", () => {
  const existsSyncMock = vi.fn(() => true);
  const readFileSyncMock = vi.fn((_path: string) =>
    JSON.stringify({ system_prompt: "You are helpful", tools: [] }),
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
  stdin.write("\r");
  await wait(50);
}

/** Press Enter */
async function pressEnter(stdin: { write: (s: string) => void }) {
  stdin.write("\r");
  await wait(50);
}

/** Press down arrow */
async function pressDown(stdin: { write: (s: string) => void }) {
  stdin.write("\u001B[B");
  await wait(30);
}

/** Press Escape */
async function pressEsc(stdin: { write: (s: string) => void }) {
  stdin.write("\u001B");
  await wait(50);
}

// ─── Import component (after mocks are set up) ────────────────────────────────

let SimulationsApp: React.ComponentType<{ onBack?: () => void }>;

// ─── Tests ────────────────────────────────────────────────────────────────────

describe("SimulationsApp", () => {
  beforeEach(async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    spawnMock.mockReset();
    spawnMock.mockImplementation(() => makeFakeProc());

    const mod = await import("../source/sim-app.js");
    SimulationsApp = mod.SimulationsApp;
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.resetModules();
  });

  // ─── Helper: render and wait for init to complete ──────────────────────────
  async function renderAndInit() {
    const { stdout, stdin } = render(<SimulationsApp onBack={() => {}} />);
    // Wait for init useEffect to fire (init → select-type)
    await wait(100);
    return { stdout, stdin };
  }

  // ─── Helper: advance to config-path with "text" type ──────────────────────
  async function reachConfigPath(type: "text" | "voice" = "text") {
    const { stdout, stdin } = await renderAndInit();
    // select-type is shown; Text is first item (default), Voice is second
    if (type === "voice") {
      await pressDown(stdin); // move to Voice
    }
    await pressEnter(stdin); // select → config-path
    return { stdout, stdin };
  }

  // ─── Helper: advance to provider step (text calibrate agent) ──────────────
  async function reachProvider() {
    const fs = await import("node:fs");
    vi.mocked(fs.readFileSync).mockReturnValue(
      JSON.stringify({ system_prompt: "You are helpful", tools: [] }),
    );
    const { stdout, stdin } = await reachConfigPath("text");
    await typeAndSubmit(stdin, "./config.json"); // → provider
    return { stdout, stdin };
  }

  // ─── Helper: advance to enter-model step ──────────────────────────────────
  async function reachEnterModel() {
    const { stdout, stdin } = await reachProvider();
    await pressEnter(stdin); // select OpenRouter → enter-model
    return { stdout, stdin };
  }

  // ─── Helper: advance to output-dir step (text calibrate agent) ────────────
  async function reachOutputDirTextCalibrate() {
    const { stdout, stdin } = await reachEnterModel();
    await pressEnter(stdin); // submit empty → uses default model → output-dir
    return { stdout, stdin };
  }

  // ─── Helper: advance to output-dir step (voice calibrate agent) ───────────
  async function reachOutputDirVoice() {
    const fs = await import("node:fs");
    vi.mocked(fs.readFileSync).mockReturnValue(
      JSON.stringify({ system_prompt: "You are helpful", tools: [] }),
    );
    const { stdout, stdin } = await reachConfigPath("voice");
    await typeAndSubmit(stdin, "./config.json"); // voice + no agent_url → output-dir
    return { stdout, stdin };
  }

  // ─── Helper: advance to output-dir step (agent connection) ────────────────
  async function reachOutputDirAgentConnection() {
    const fs = await import("node:fs");
    vi.mocked(fs.readFileSync).mockReturnValue(
      JSON.stringify({ agent_url: "http://localhost:8080" }),
    );
    const { stdout, stdin } = await reachConfigPath("text");
    await typeAndSubmit(stdin, "./config.json"); // text + agent_url → output-dir
    return { stdout, stdin };
  }

  // ─── Helper: advance to parallel step (text calibrate agent) ──────────────
  async function reachParallel() {
    const { stdout, stdin } = await reachOutputDirTextCalibrate();
    await pressEnter(stdin); // submit output-dir → parallel
    return { stdout, stdin };
  }

  // ─── Helper: advance to api-keys step ────────────────────────────────────
  async function reachApiKeys() {
    // Make getCredential return null to force api-keys step
    const credentials = await import("../source/credentials.js");
    vi.mocked(credentials.getCredential).mockReturnValue(null as unknown as string);

    const { stdout, stdin } = await reachParallel();
    await pressEnter(stdin); // submit parallel → api-keys (keys missing)
    return { stdout, stdin };
  }

  // ─── Helper: get to running step for text calibrate agent ─────────────────
  async function reachRunningTextCalibrate(modelName = "gpt-4.1") {
    const fs = await import("node:fs");
    vi.mocked(fs.readFileSync).mockReturnValue(
      JSON.stringify({ system_prompt: "You are helpful", tools: [] }),
    );
    vi.mocked(fs.readdirSync).mockReturnValue([]);

    const credentials = await import("../source/credentials.js");
    vi.mocked(credentials.getCredential).mockReturnValue("sk-fake-key");

    const proc = makeFakeProc();
    spawnMock.mockImplementation(() => proc);

    const { stdout, stdin } = await reachConfigPath("text");
    await typeAndSubmit(stdin, "./config.json"); // → provider
    await pressEnter(stdin); // select OpenRouter → enter-model
    await typeAndSubmit(stdin, modelName); // → output-dir
    await pressEnter(stdin); // submit output-dir → parallel
    await pressEnter(stdin); // submit parallel (keys present) → running

    await wait(100);
    return { stdout, stdin, proc };
  }

  // ─── Helper: get to running step for voice calibrate agent ────────────────
  async function reachRunningVoiceCalibrate() {
    const fs = await import("node:fs");
    vi.mocked(fs.readFileSync).mockReturnValue(
      JSON.stringify({ system_prompt: "You are helpful", tools: [] }),
    );
    vi.mocked(fs.readdirSync).mockReturnValue([]);

    const credentials = await import("../source/credentials.js");
    vi.mocked(credentials.getCredential).mockReturnValue("sk-fake-key");

    const proc = makeFakeProc();
    spawnMock.mockImplementation(() => proc);

    const { stdout, stdin } = await reachConfigPath("voice");
    await typeAndSubmit(stdin, "./config.json"); // voice + no agent_url → output-dir
    await pressEnter(stdin); // submit output-dir → parallel
    await pressEnter(stdin); // submit parallel (keys present) → running

    await wait(100);
    return { stdout, stdin, proc };
  }

  // ─── Helper: get to running step for agent connection (text) ──────────────
  async function reachRunningAgentConnection() {
    const fs = await import("node:fs");
    vi.mocked(fs.readFileSync).mockReturnValue(
      JSON.stringify({ agent_url: "http://localhost:8080" }),
    );
    vi.mocked(fs.readdirSync).mockReturnValue([]);

    const credentials = await import("../source/credentials.js");
    vi.mocked(credentials.getCredential).mockReturnValue("sk-fake-key");

    const proc = makeFakeProc();
    spawnMock.mockImplementation(() => proc);

    const { stdout, stdin } = await reachConfigPath("text");
    await typeAndSubmit(stdin, "./config.json"); // text + agent_url → output-dir
    await pressEnter(stdin); // submit output-dir → parallel
    await pressEnter(stdin); // submit parallel (keys present) → running

    await wait(100);
    return { stdout, stdin, proc };
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Step rendering tests
  // ──────────────────────────────────────────────────────────────────────────

  describe("step rendering", () => {
    it("shows select-type step on init", async () => {
      const { stdout } = await renderAndInit();
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Text");
      expect(frame).toContain("Voice");
    });

    it("shows config-path step after selecting text", async () => {
      const { stdout } = await reachConfigPath("text");
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Config file");
    });

    it("shows config-path step after selecting voice", async () => {
      const { stdout } = await reachConfigPath("voice");
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Config file");
    });

    it("shows provider step for text calibrate agent config", async () => {
      const { stdout } = await reachProvider();
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Provider");
    });

    it("shows enter-model step after selecting provider", async () => {
      const { stdout } = await reachEnterModel();
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Model");
    });

    it("shows output-dir step", async () => {
      const { stdout } = await reachOutputDirTextCalibrate();
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Output directory");
    });

    it("shows parallel step after output-dir", async () => {
      const { stdout } = await reachParallel();
      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Parallel");
    });

    it("shows api-keys step when keys are missing", async () => {
      const { stdout } = await reachApiKeys();
      const frame = stdout.lastFrame() ?? "";
      // api-keys step shows the key name as the prompt
      expect(frame).toMatch(/API_KEY|sk-/i);
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // Agent connection flow tests
  // ──────────────────────────────────────────────────────────────────────────

  describe("agent connection flow", () => {
    it("text + agent_url config skips provider and goes to output-dir", async () => {
      const fs = await import("node:fs");
      vi.mocked(fs.readFileSync).mockReturnValue(
        JSON.stringify({ agent_url: "http://localhost:8080" }),
      );

      const { stdout, stdin } = await reachConfigPath("text");
      await typeAndSubmit(stdin, "./config.json");

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Output directory");
      expect(frame).not.toContain("Provider");
    });

    it("text + calibrate agent config goes to provider", async () => {
      const fs = await import("node:fs");
      vi.mocked(fs.readFileSync).mockReturnValue(
        JSON.stringify({ system_prompt: "You are helpful", tools: [] }),
      );

      const { stdout, stdin } = await reachConfigPath("text");
      await typeAndSubmit(stdin, "./config.json");

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Provider");
    });

    it("voice + agent_url config shows error on config-path step", async () => {
      const fs = await import("node:fs");
      vi.mocked(fs.readFileSync).mockReturnValue(
        JSON.stringify({ agent_url: "http://localhost:8080" }),
      );

      const { stdout, stdin } = await reachConfigPath("voice");
      await typeAndSubmit(stdin, "./config.json");

      const frame = stdout.lastFrame() ?? "";
      // Should still be on config-path (not output-dir) and show an error
      expect(frame).toContain("Config file");
      expect(frame).toContain("not supported");
    });

    it("error message clears when user starts typing", async () => {
      const fs = await import("node:fs");
      vi.mocked(fs.readFileSync).mockReturnValue(
        JSON.stringify({ agent_url: "http://localhost:8080" }),
      );

      const { stdout, stdin } = await reachConfigPath("voice");
      await typeAndSubmit(stdin, "./config.json"); // triggers error

      let frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("not supported");

      // Now type a character — onChange clears the error via setInitError("")
      stdin.write("x");
      await wait(50);

      frame = stdout.lastFrame() ?? "";
      expect(frame).not.toContain("not supported");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // Back navigation tests
  // ──────────────────────────────────────────────────────────────────────────

  describe("back navigation", () => {
    it("select-type esc calls onBack", async () => {
      const onBack = vi.fn();
      const { stdin } = render(<SimulationsApp onBack={onBack} />);
      await wait(100); // init → select-type

      await pressEsc(stdin);

      expect(onBack).toHaveBeenCalled();
    });

    it("config-path esc goes to select-type", async () => {
      const { stdout, stdin } = await reachConfigPath("text");

      await pressEsc(stdin);

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Text");
      expect(frame).toContain("Voice");
    });

    it("provider esc goes to config-path", async () => {
      const { stdout, stdin } = await reachProvider();

      await pressEsc(stdin);

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Config file");
    });

    it("enter-model esc goes to provider", async () => {
      const { stdout, stdin } = await reachEnterModel();

      await pressEsc(stdin);

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Provider");
    });

    it("output-dir esc (text calibrate agent) goes to enter-model", async () => {
      const { stdout, stdin } = await reachOutputDirTextCalibrate();

      await pressEsc(stdin);

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Model");
    });

    it("output-dir esc (voice) goes to config-path", async () => {
      const { stdout, stdin } = await reachOutputDirVoice();

      await pressEsc(stdin);

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Config file");
    });

    it("output-dir esc (agent connection) goes to config-path", async () => {
      const { stdout, stdin } = await reachOutputDirAgentConnection();

      await pressEsc(stdin);

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Config file");
    });

    it("parallel esc goes to output-dir", async () => {
      const { stdout, stdin } = await reachParallel();

      await pressEsc(stdin);

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Output directory");
    });

    it("api-keys esc goes to parallel", async () => {
      const { stdout, stdin } = await reachApiKeys();

      await pressEsc(stdin);

      const frame = stdout.lastFrame() ?? "";
      expect(frame).toContain("Parallel");
    });
  });

  // ──────────────────────────────────────────────────────────────────────────
  // CMD args tests
  // ──────────────────────────────────────────────────────────────────────────

  describe("startSimulation cmdArgs", () => {
    it("text calibrate agent includes --type text -m model -p provider", async () => {
      const { proc: _proc } = await reachRunningTextCalibrate("gpt-4.1");
      await wait(50);

      const calls = spawnMock.mock.calls;
      const simCall = calls.find(
        (c) => Array.isArray(c[1]) && c[1].includes("simulations"),
      );
      expect(simCall).toBeDefined();
      const args = simCall![1] as string[];
      expect(args).toContain("simulations");
      expect(args).toContain("--type");
      expect(args).toContain("text");
      expect(args).toContain("-m");
      expect(args).toContain("gpt-4.1");
      expect(args).toContain("-p");
      expect(args).toContain("openrouter");
    });

    it("voice calibrate agent includes --type voice, no -m or -p", async () => {
      await reachRunningVoiceCalibrate();
      await wait(50);

      const calls = spawnMock.mock.calls;
      const simCall = calls.find(
        (c) => Array.isArray(c[1]) && c[1].includes("simulations"),
      );
      expect(simCall).toBeDefined();
      const args = simCall![1] as string[];
      expect(args).toContain("simulations");
      expect(args).toContain("--type");
      expect(args).toContain("voice");
      expect(args).not.toContain("-m");
      expect(args).not.toContain("-p");
    });

    it("agent connection includes --type text with default model and provider", async () => {
      await reachRunningAgentConnection();
      await wait(50);

      const calls = spawnMock.mock.calls;
      const simCall = calls.find(
        (c) => Array.isArray(c[1]) && c[1].includes("simulations"),
      );
      expect(simCall).toBeDefined();
      const args = simCall![1] as string[];
      expect(args).toContain("simulations");
      expect(args).toContain("--type");
      expect(args).toContain("text");
      // Default model and provider since provider/enter-model steps were skipped
      expect(args).toContain("-m");
      expect(args).toContain("gpt-4.1");
      expect(args).toContain("-p");
      expect(args).toContain("openrouter");
    });

    it("parallel > 1 includes -n flag", async () => {
      const fs = await import("node:fs");
      vi.mocked(fs.readFileSync).mockReturnValue(
        JSON.stringify({ system_prompt: "You are helpful", tools: [] }),
      );
      vi.mocked(fs.readdirSync).mockReturnValue([]);

      const credentials = await import("../source/credentials.js");
      vi.mocked(credentials.getCredential).mockReturnValue("sk-fake-key");

      const proc = makeFakeProc();
      spawnMock.mockImplementation(() => proc);

      const { stdin } = await reachConfigPath("text");
      await typeAndSubmit(stdin, "./config.json"); // → provider
      await pressEnter(stdin); // → enter-model
      await pressEnter(stdin); // → output-dir
      await pressEnter(stdin); // submit output-dir → parallel

      // Type parallel count — parallelInput starts as "1", typing "3" appends → "13"
      // parseInt("13") = 13 > 1 so -n flag is included
      await typeAndSubmit(stdin, "3"); // → running (keys present)
      await wait(100);

      const calls = spawnMock.mock.calls;
      const simCall = calls.find(
        (c) => Array.isArray(c[1]) && c[1].includes("simulations"),
      );
      expect(simCall).toBeDefined();
      const args = simCall![1] as string[];
      expect(args).toContain("-n");
      // The -n value should be > 1 (parallel count is "13" since "1" is the default + "3" appended)
      const nIdx = args.indexOf("-n");
      expect(parseInt(args[nIdx + 1]!)).toBeGreaterThan(1);
    });
  });
});
