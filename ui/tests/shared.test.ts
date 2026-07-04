import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the node built-ins that findCalibrateBin touches so we can drive each
// discovery branch without a real environment.
const execSync = vi.fn();
const existsSync = vi.fn();

vi.mock("node:child_process", () => ({ execSync: (...a: unknown[]) => execSync(...a) }));
vi.mock("node:fs", () => ({ default: { existsSync: (...a: unknown[]) => existsSync(...a) } }));

import { findCalibrateBin } from "../source/shared.js";

beforeEach(() => {
  execSync.mockReset();
  existsSync.mockReset();
  execSync.mockImplementation(() => {
    throw new Error("not found");
  });
  existsSync.mockReturnValue(false);
});

describe("findCalibrateBin", () => {
  it("prefers the new `calibrate-agent` name on PATH", () => {
    execSync.mockImplementation((cmd: string) => {
      if (cmd === "which calibrate-agent") return Buffer.from("/usr/bin/calibrate-agent");
      throw new Error("not found");
    });
    expect(findCalibrateBin()).toEqual({ cmd: "calibrate-agent", args: [] });
  });

  it("falls back to the legacy `calibrate` name when `calibrate-agent` is absent", () => {
    execSync.mockImplementation((cmd: string) => {
      if (cmd === "which calibrate") return Buffer.from("/usr/bin/calibrate");
      throw new Error("not found");
    });
    expect(findCalibrateBin()).toEqual({ cmd: "calibrate", args: [] });
  });

  it("finds `calibrate-agent` in a local .venv when not on PATH", () => {
    existsSync.mockImplementation((p: string) => p.endsWith("/.venv/bin/calibrate-agent"));
    const result = findCalibrateBin();
    expect(result?.cmd).toMatch(/\/\.venv\/bin\/calibrate-agent$/);
    expect(result?.args).toEqual([]);
  });

  it("falls back to `uv run calibrate-agent` when nothing else is found", () => {
    execSync.mockImplementation((cmd: string) => {
      if (cmd === "uv run which calibrate-agent") return Buffer.from("ok");
      throw new Error("not found");
    });
    expect(findCalibrateBin()).toEqual({ cmd: "uv", args: ["run", "calibrate-agent"] });
  });

  it("returns null when neither name can be resolved anywhere", () => {
    expect(findCalibrateBin()).toBeNull();
  });
});
