import React, { useState, useEffect, useRef } from "react";
import { Box, Text, useApp, useInput } from "ink";
import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import {
  SelectInput,
  TextInput,
  Spinner,
  Table,
  BarChart,
} from "./components.js";
import { getCredential, saveCredential } from "./credentials.js";
import { type CalibrateCmd, findCalibrateBin, stripAnsi } from "./shared.js";

// ─── Types ───────────────────────────────────────────────────
type SimStep =
  | "init"
  | "select-type"
  | "config-path"
  | "provider"
  | "enter-model"
  | "parallel"
  | "output-dir"
  | "output-dir-confirm"
  | "api-keys"
  | "running"
  | "leaderboard";

interface SimConfig {
  type: "text" | "voice";
  configPath: string;
  models: string[];
  provider: string;
  outputDir: string;
  parallel: number;
  overwrite: boolean;
  envVars: Record<string, string>;
  calibrate: CalibrateCmd;
}

interface ModelState {
  status: "waiting" | "running" | "done" | "error";
  logs: string[];
  metrics?: Record<string, number>;
}

interface SimSlotState {
  name: string; // e.g., "simulation_persona_1_scenario_1"
  personaIdx: number;
  scenarioIdx: number;
  logs: string[];
  status: "running" | "done";
}

interface SimulationResult {
  persona_idx: number;
  scenario_idx: number;
  name: string;
  value: number;
  reasoning: string;
}

interface EvalResult {
  simulation: string;
  persona_idx: number;
  scenario_idx: number;
  criteria: { name: string; value: number; reasoning: string }[];
}

// ─── Model Examples ───────────────────────────────────────────
const OPENAI_MODEL_EXAMPLES = [
  "gpt-4.1",
  "gpt-4.1-mini",
  "gpt-4o",
  "gpt-4o-mini",
  "o1",
  "o1-mini",
  "o3-mini",
];

const OPENROUTER_MODEL_EXAMPLES = [
  "openai/gpt-4.1",
  "anthropic/claude-sonnet-4",
  "google/gemini-2.0-flash-001",
];

const MAX_PARALLEL_MODELS = 2;

// ─── Main Component ──────────────────────────────────────────
export function SimulationsApp({ onBack }: { onBack?: () => void }) {
  const { exit } = useApp();
  const [step, setStep] = useState<SimStep>("init");
  const [config, setConfig] = useState<SimConfig>({
    type: "text",
    configPath: "",
    models: [],
    provider: "openrouter",
    outputDir: "./out",
    parallel: 1,
    overwrite: false,
    envVars: {},
    calibrate: { cmd: "calibrate", args: [] },
  });

  // ── overwrite confirmation state ──
  const [existingDirs, setExistingDirs] = useState<string[]>([]);

  // ── input state ──
  const [configInput, setConfigInput] = useState("");
  const [outputInput, setOutputInput] = useState("./out");
  const [parallelInput, setParallelInput] = useState("1");
  const [modelInput, setModelInput] = useState("");

  // ── API key state ──
  const [missingKeys, setMissingKeys] = useState<string[]>([]);
  const [currentKeyIdx, setCurrentKeyIdx] = useState(0);
  const [keyInput, setKeyInput] = useState("");

  // ── run state (multi-model) ──
  const [modelStates, setModelStates] = useState<Record<string, ModelState>>(
    {}
  );
  const [phase, setPhase] = useState<"eval" | "leaderboard" | "done">("eval");
  const [runningCount, setRunningCount] = useState(0);
  const [nextModelIdx, setNextModelIdx] = useState(0);
  const processRefs = useRef<Map<string, ChildProcess>>(new Map());
  const calibrateBin = useRef<CalibrateCmd | null>(null);

  // ── simulation slot state (for text simulations) ──
  const [simSlots, setSimSlots] = useState<SimSlotState[]>([]);
  const [simProcessRunning, setSimProcessRunning] = useState(false);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // ── init error state ──
  const [initError, setInitError] = useState("");

  // ── leaderboard state ──
  const [view, setView] = useState<"leaderboard" | "sim-detail">("leaderboard");
  const [selectedSim, setSelectedSim] = useState<string | null>(null);
  const [evalResults, setEvalResults] = useState<EvalResult[]>([]);
  const [scrollOffset, setScrollOffset] = useState(0);
  const MAX_VISIBLE_ROWS = 10;

  const [metrics, setMetrics] = useState<
    Record<string, { mean: number; std: number; values: number[] }>
  >({});

  // Step navigation helper
  const goBack = () => {
    switch (step) {
      case "select-type":
        if (onBack) onBack();
        break;
      case "config-path":
        setStep("select-type");
        break;
      case "provider":
        setStep("config-path");
        break;
      case "enter-model":
        setStep("provider");
        setModelInput("");
        break;
      case "parallel":
        setStep("enter-model");
        break;
      case "output-dir":
        if (config.type === "text") {
          setStep("parallel");
        } else {
          setStep("config-path");
        }
        break;
      case "output-dir-confirm":
        setStep("output-dir");
        setExistingDirs([]);
        break;
      case "api-keys":
        setStep("output-dir");
        setCurrentKeyIdx(0);
        setKeyInput("");
        break;
    }
  };

  // Check for existing output directories
  const checkExistingOutput = (outputDir: string): string[] => {
    const existing: string[] = [];
    try {
      if (!fs.existsSync(outputDir)) return [];

      const entries = fs.readdirSync(outputDir, { withFileTypes: true });
      for (const entry of entries) {
        if (entry.isDirectory()) {
          const dirPath = path.join(outputDir, entry.name);
          try {
            const contents = fs.readdirSync(dirPath);
            if (contents.length > 0) {
              existing.push(entry.name);
            }
          } catch {
            // Ignore read errors
          }
        } else if (
          entry.isFile() &&
          (entry.name === "metrics.json" || entry.name === "results.csv")
        ) {
          existing.push(entry.name);
        }
      }
    } catch {
      // Output dir doesn't exist yet, that's fine
    }
    return existing;
  };

  useInput((input, key) => {
    if (input === "q") {
      if (step === "leaderboard") {
        if (view === "sim-detail") {
          setView("leaderboard");
          setSelectedSim(null);
          setScrollOffset(0);
        } else {
          if (onBack) onBack();
          else exit();
        }
      }
    }
    if (input === "b" && step === "init" && onBack) onBack();
    // Escape key to go back to previous step
    if (key.escape) {
      if (step === "leaderboard" && view === "sim-detail") {
        setView("leaderboard");
        setSelectedSim(null);
        setScrollOffset(0);
      } else if (!["init", "running", "leaderboard"].includes(step)) {
        goBack();
      }
    }
    // Scroll in detail view
    if (step === "leaderboard" && view === "sim-detail") {
      const selectedResult = evalResults.find((r) => r.simulation === selectedSim);
      const itemCount = selectedResult?.criteria.length || 0;
      if (key.upArrow && scrollOffset > 0) {
        setScrollOffset((o) => o - 1);
      }
      if (key.downArrow && scrollOffset < itemCount - MAX_VISIBLE_ROWS) {
        setScrollOffset((o) => o + 1);
      }
    }
  });

  // ── Init ──
  useEffect(() => {
    if (step !== "init") return;
    calibrateBin.current = findCalibrateBin();
    if (!calibrateBin.current) {
      setInitError("Error: calibrate binary not found");
      setStep("leaderboard");
      return;
    }
    setConfig((c) => ({ ...c, calibrate: calibrateBin.current! }));
    setStep("select-type");
  }, [step]);

  // ── Check API keys ──
  function checkApiKeys(simType: string, provider: string) {
    const needed: string[] = [];
    if (simType === "text") {
      // Always need OPENAI_API_KEY for LLM judge/evaluation
      if (!getCredential("OPENAI_API_KEY")) {
        needed.push("OPENAI_API_KEY");
      }
      // Need OPENROUTER_API_KEY if using OpenRouter
      if (provider === "openrouter") {
        if (!getCredential("OPENROUTER_API_KEY")) {
          needed.push("OPENROUTER_API_KEY");
        }
      }
    }
    // Voice simulations need keys based on config file providers — hard to check here.
    return needed;
  }

  // ── Build model directory name ──
  function getModelDir(model: string): string {
    let modelDir =
      config.provider === "openai" ? `${config.provider}/${model}` : model;
    return modelDir.replace(/\//g, "__");
  }

  // ── Initialize states when entering running step ──
  useEffect(() => {
    if (step !== "running") return;

    // For voice simulations, we run a single process
    if (config.type === "voice") {
      setModelStates({ voice: { status: "running", logs: [] } });
      setPhase("eval");
      return;
    }

    // For text simulations, run a single process and poll for simulation directories
    setSimSlots([]);
    setSimProcessRunning(true);
    setPhase("eval");
  }, [step]);

  // ── Poll simulation directories for logs (text simulations) ──
  const pollSimulationDirs = () => {
    try {
      if (!fs.existsSync(config.outputDir)) return;

      const entries = fs.readdirSync(config.outputDir, { withFileTypes: true });
      const simDirs = entries
        .filter(
          (e) => e.isDirectory() && e.name.startsWith("simulation_persona_")
        )
        .map((e) => e.name);

      const newSlots: SimSlotState[] = [];

      for (const dirName of simDirs) {
        // Parse persona and scenario indices from directory name
        const match = dirName.match(/simulation_persona_(\d+)_scenario_(\d+)/);
        if (!match) continue;

        const personaIdx = parseInt(match[1]!, 10);
        const scenarioIdx = parseInt(match[2]!, 10);

        // Read results.log if it exists
        const logPath = path.join(config.outputDir, dirName, "results.log");
        let logs: string[] = [];
        let status: "running" | "done" = "running";

        if (fs.existsSync(logPath)) {
          try {
            const content = fs.readFileSync(logPath, "utf-8");
            logs = content
              .split("\n")
              .filter((l) => l.trim())
              .slice(-15);
          } catch {
            // Ignore read errors
          }
        }

        // Check if evaluation_results.csv exists (indicates completion)
        const evalPath = path.join(
          config.outputDir,
          dirName,
          "evaluation_results.csv"
        );
        if (fs.existsSync(evalPath)) {
          status = "done";
        }

        newSlots.push({
          name: dirName,
          personaIdx,
          scenarioIdx,
          logs,
          status,
        });
      }

      // Sort by persona then scenario
      newSlots.sort((a, b) => {
        if (a.personaIdx !== b.personaIdx)
          return a.personaIdx - b.personaIdx;
        return a.scenarioIdx - b.scenarioIdx;
      });

      setSimSlots(newSlots);
    } catch {
      // Ignore errors
    }
  };

  // ── Run text simulation (single process with directory polling) ──
  useEffect(() => {
    if (step !== "running" || config.type !== "text" || !config.calibrate)
      return;
    if (!simProcessRunning) return;

    const bin = config.calibrate;
    const env: Record<string, string> = { ...process.env } as Record<
      string,
      string
    >;

    for (const k of ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]) {
      const v = getCredential(k);
      if (v) env[k] = v;
    }
    Object.assign(env, config.envVars);
    env.PYTHONUNBUFFERED = "1";

    const model = config.models[0] || "gpt-4.1";
    const cmdArgs = [
      ...bin.args,
      "simulations",
      "--type",
      "text",
      "-c",
      config.configPath,
      "-o",
      config.outputDir,
      "-m",
      model,
      "-p",
      config.provider,
    ];

    if (config.parallel > 1) {
      cmdArgs.push("-n", String(config.parallel));
    }

    const proc = spawn(bin.cmd, cmdArgs, {
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });

    processRefs.current.set("text-sim", proc);

    // Start polling for simulation directories
    pollingRef.current = setInterval(pollSimulationDirs, 500);

    proc.on("close", (code) => {
      // Stop polling
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }

      // Final poll to get latest state
      pollSimulationDirs();

      setSimProcessRunning(false);
      processRefs.current.delete("text-sim");

      if (code === 0) {
        // Load results and show leaderboard
        loadMetrics();
        loadEvalResults();
        setPhase("done");
        setTimeout(() => setStep("leaderboard"), 500);
      } else {
        setPhase("done");
        setTimeout(() => setStep("leaderboard"), 500);
      }
    });

    proc.on("error", () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      setSimProcessRunning(false);
      processRefs.current.delete("text-sim");
      setPhase("done");
      setTimeout(() => setStep("leaderboard"), 500);
    });

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      proc.kill();
    };
  }, [step, config.type, simProcessRunning]);

  // ── Run voice simulation (single process) ──
  useEffect(() => {
    if (step !== "running" || config.type !== "voice" || !config.calibrate)
      return;

    const bin = config.calibrate;
    const env: Record<string, string> = { ...process.env } as Record<
      string,
      string
    >;

    for (const k of ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]) {
      const v = getCredential(k);
      if (v) env[k] = v;
    }
    Object.assign(env, config.envVars);
    env.PYTHONUNBUFFERED = "1";

    const cmdArgs = [
      ...bin.args,
      "simulations",
      "--type",
      "voice",
      "-c",
      config.configPath,
      "-o",
      config.outputDir,
    ];

    const proc = spawn(bin.cmd, cmdArgs, {
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });

    processRefs.current.set("voice", proc);

    const onData = (data: Buffer) => {
      const lines = data
        .toString()
        .split(/[\r\n]+/)
        .filter((l) => l.trim());
      setModelStates((prev) => ({
        ...prev,
        voice: {
          ...prev.voice!,
          logs: [...prev.voice!.logs, ...lines].slice(-20),
        },
      }));
    };

    proc.stdout?.on("data", onData);
    proc.stderr?.on("data", onData);

    proc.on("close", (code) => {
      setModelStates((prev) => ({
        ...prev,
        voice: { ...prev.voice!, status: code === 0 ? "done" : "error" },
      }));
      processRefs.current.delete("voice");

      // Load results
      loadMetrics();
      loadEvalResults();
      setPhase("done");
      setTimeout(() => setStep("leaderboard"), 500);
    });

    return () => {
      proc.kill();
    };
  }, [step, config.type]);

  // ── Load metrics for leaderboard ──
  const loadMetrics = () => {
    const results: typeof metrics = [];

    if (config.type === "voice") {
      // Voice simulation has a single output
      try {
        const metricsPath = path.join(config.outputDir, "metrics.json");
        if (fs.existsSync(metricsPath)) {
          const raw = JSON.parse(fs.readFileSync(metricsPath, "utf-8"));
          const entry: (typeof metrics)[0] = { model: "voice" };
          for (const [key, val] of Object.entries(raw)) {
            if (typeof val === "object" && val !== null && "mean" in val) {
              entry[key] = (val as { mean: number }).mean;
            } else if (typeof val === "number") {
              entry[key] = val;
            }
          }
          results.push(entry);
        }
      } catch {
        // Ignore
      }
    } else {
      // Text simulations with multiple models
      for (const model of config.models) {
        try {
          const modelDir = getModelDir(model);
          const metricsPath = path.join(
            config.outputDir,
            modelDir,
            "metrics.json"
          );
          if (fs.existsSync(metricsPath)) {
            const raw = JSON.parse(fs.readFileSync(metricsPath, "utf-8"));
            const entry: (typeof metrics)[0] = { model };
            for (const [key, val] of Object.entries(raw)) {
              if (typeof val === "object" && val !== null && "mean" in val) {
                entry[key] = (val as { mean: number }).mean;
              } else if (typeof val === "number") {
                entry[key] = val;
              }
            }
            results.push(entry);
          }
        } catch {
          // Skip models with no results
        }
      }
    }

    setMetrics(results);
  };

  // ── Load model results when selected ──
  useEffect(() => {
    if (!selectedModel) return;
    try {
      let resultsPath: string;
      if (config.type === "voice") {
        resultsPath = path.join(config.outputDir, "results.csv");
      } else {
        const modelDir = getModelDir(selectedModel);
        resultsPath = path.join(config.outputDir, modelDir, "results.csv");
      }

      if (fs.existsSync(resultsPath)) {
        const csvContent = fs.readFileSync(resultsPath, "utf-8");
        const lines = csvContent.trim().split("\n");
        if (lines.length < 2) {
          setModelResults([]);
          return;
        }
        const headers = lines[0]!.split(",").map((h) => h.trim());
        const results: SimulationResult[] = [];
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i]!.split(",");
          const row: SimulationResult = { persona_idx: 0, scenario_idx: 0 };
          headers.forEach((h, idx) => {
            const val = values[idx]?.trim() || "";
            const num = parseFloat(val);
            row[h] = isNaN(num) ? val : num;
          });
          results.push(row);
        }
        setModelResults(results);
        setScrollOffset(0);
      } else {
        setModelResults([]);
      }
    } catch {
      setModelResults([]);
    }
  }, [selectedModel, config.outputDir, config.type]);

  // ── Cleanup on unmount ──
  useEffect(() => {
    return () => {
      processRefs.current.forEach((proc) => proc.kill());
    };
  }, []);

  // ── Render ──
  const header = (
    <Box marginBottom={1}>
      <Text bold color="cyan">
        Simulations
      </Text>
    </Box>
  );

  switch (step) {
    case "init":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Spinner label="Initializing..." />
        </Box>
      );

    case "select-type":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            Text simulations use LLM-only conversations. Voice simulations use
            the full STT → LLM → TTS pipeline.
          </Text>
          <Text>Simulation type:</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "Text simulation", value: "text" },
                { label: "Voice simulation", value: "voice" },
              ]}
              onSelect={(v) => {
                setConfig((c) => ({ ...c, type: v as "text" | "voice" }));
                setStep("config-path");
              }}
            />
          </Box>
          {onBack && (
            <Box marginTop={1}>
              <Text dimColor>Press Esc to go back</Text>
            </Box>
          )}
        </Box>
      );

    case "config-path":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            Path to a JSON config file containing system prompt, tools,
            personas, scenarios, and evaluation criteria.
          </Text>
          <Box marginTop={1}>
            <Text>Config file: </Text>
            <TextInput
              value={configInput}
              onChange={setConfigInput}
              onSubmit={(v) => {
                if (v.trim()) {
                  const resolved = path.resolve(v.trim());
                  if (!fs.existsSync(resolved)) {
                    setConfigInput("");
                    return;
                  }
                  setConfig((c) => ({ ...c, configPath: resolved }));
                  if (config.type === "text") {
                    setStep("provider");
                  } else {
                    setStep("output-dir");
                  }
                }
              }}
              placeholder="./config.json"
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>Enter to submit, Esc to go back</Text>
          </Box>
        </Box>
      );

    case "provider":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>Which LLM provider to use for the simulation.</Text>
          <Text>Provider:</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "OpenRouter", value: "openrouter" },
                { label: "OpenAI", value: "openai" },
              ]}
              onSelect={(v) => {
                setConfig((c) => ({ ...c, provider: v, models: [] }));
                setModelInput("");
                setStep("enter-model");
              }}
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>Press Esc to go back</Text>
          </Box>
        </Box>
      );

    case "enter-model": {
      const modelExamples =
        config.provider === "openai"
          ? OPENAI_MODEL_EXAMPLES
          : OPENROUTER_MODEL_EXAMPLES;
      const defaultModel =
        config.provider === "openai" ? "gpt-4.1" : "openai/gpt-4.1";
      const platformHint =
        config.provider === "openai"
          ? "Enter model name exactly as it appears on OpenAI (platform.openai.com)"
          : "Enter model name exactly as it appears on OpenRouter (openrouter.ai/models)";

      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Box marginBottom={1}>
            <Text dimColor>Provider: {config.provider}</Text>
          </Box>
          <Text dimColor>{platformHint}</Text>
          <Box marginTop={1} flexDirection="column">
            <Text dimColor>Examples: {modelExamples.join(", ")}</Text>
          </Box>
          <Box marginTop={1}>
            <Text>Model: </Text>
            <TextInput
              value={modelInput}
              onChange={setModelInput}
              onSubmit={(v) => {
                const trimmed = v.trim();
                const modelToUse = trimmed || defaultModel;
                setConfig((c) => ({ ...c, models: [modelToUse] }));
                setModelInput("");
                setStep("parallel");
              }}
              placeholder={defaultModel}
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>
              Enter to submit (default: {defaultModel}), Esc to go back
            </Text>
          </Box>
        </Box>
      );
    }

    case "parallel":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            Run multiple persona × scenario combinations at the same time to
            speed things up.
          </Text>
          <Box marginTop={1}>
            <Text>Parallel simulations per model: </Text>
            <TextInput
              value={parallelInput}
              onChange={setParallelInput}
              onSubmit={(v) => {
                setConfig((c) => ({
                  ...c,
                  parallel: parseInt(v) || 1,
                }));
                setStep("output-dir");
              }}
              placeholder="1"
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>Enter to submit (default: 1), Esc to go back</Text>
          </Box>
        </Box>
      );

    case "output-dir":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            Directory where simulation results will be saved.
          </Text>
          <Box marginTop={1}>
            <Text>Output directory: </Text>
            <TextInput
              value={outputInput}
              onChange={setOutputInput}
              onSubmit={(v) => {
                const trimmed = v.trim() || "./out";
                setConfig((c) => ({ ...c, outputDir: trimmed }));

                // Check for existing data
                const existing = checkExistingOutput(trimmed);
                if (existing.length > 0) {
                  setExistingDirs(existing);
                  setStep("output-dir-confirm");
                  return;
                }

                // No existing data, proceed
                const missing = checkApiKeys(config.type, config.provider);
                if (missing.length > 0) {
                  setMissingKeys(missing);
                  setCurrentKeyIdx(0);
                  setStep("api-keys");
                } else {
                  setStep("running");
                }
              }}
              placeholder="./out"
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>
              Enter to submit (default: ./out), Esc to go back
            </Text>
          </Box>
        </Box>
      );

    case "output-dir-confirm":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Box marginBottom={1}>
            <Text color="yellow" bold>
              ⚠ Existing data found
            </Text>
          </Box>
          <Text>
            The following items already exist in the output directory:
          </Text>
          <Box flexDirection="column" marginLeft={2} marginY={1}>
            {existingDirs.slice(0, 5).map((dir) => (
              <Text key={dir} color="yellow">
                • {path.join(config.outputDir, dir)}
              </Text>
            ))}
            {existingDirs.length > 5 && (
              <Text dimColor>... and {existingDirs.length - 5} more</Text>
            )}
          </Box>
          <Text>Do you want to overwrite existing results?</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "Yes, overwrite and continue", value: "yes" },
                { label: "No, enter a different path", value: "no" },
              ]}
              onSelect={(v) => {
                if (v === "yes") {
                  setConfig((c) => ({ ...c, overwrite: true }));
                  const missing = checkApiKeys(config.type, config.provider);
                  if (missing.length > 0) {
                    setMissingKeys(missing);
                    setCurrentKeyIdx(0);
                    setStep("api-keys");
                  } else {
                    setStep("running");
                  }
                } else {
                  setOutputInput("");
                  setExistingDirs([]);
                  setStep("output-dir");
                }
              }}
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>Press Esc to go back</Text>
          </Box>
        </Box>
      );

    case "api-keys": {
      const currentKey = missingKeys[currentKeyIdx]!;
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>API key required for your chosen provider.</Text>
          <Box marginTop={1}>
            <Text>{currentKey}: </Text>
            <TextInput
              value={keyInput}
              onChange={setKeyInput}
              onSubmit={(v) => {
                if (v.trim()) {
                  saveCredential(currentKey, v.trim());
                  setConfig((c) => ({
                    ...c,
                    envVars: { ...c.envVars, [currentKey]: v.trim() },
                  }));
                  setKeyInput("");
                  if (currentKeyIdx + 1 < missingKeys.length) {
                    setCurrentKeyIdx(currentKeyIdx + 1);
                  } else {
                    setStep("running");
                  }
                }
              }}
              placeholder="sk-..."
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>Enter to submit, Esc to go back</Text>
          </Box>
        </Box>
      );
    }

    case "running": {
      const modelList = config.type === "voice" ? ["voice"] : config.models;
      const completedCount = Object.values(modelStates).filter(
        (s) => s.status === "done" || s.status === "error"
      ).length;

      // Get currently running models for log display
      const runningModels = modelList.filter(
        (m) => modelStates[m]?.status === "running"
      );

      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Box marginBottom={1}>
            <Text dimColor>
              Type: {config.type} | Config: {config.configPath}
            </Text>
          </Box>
          {config.type === "text" && (
            <Box marginBottom={1}>
              <Text dimColor>
                {completedCount}/{config.models.length} models
                {runningCount > 1 && ` (${runningCount} running in parallel)`}
                {" | "}Provider: {config.provider}
              </Text>
            </Box>
          )}

          {/* Model status list */}
          {modelList.map((model) => {
            const state = modelStates[model];
            if (!state) return null;
            return (
              <Box key={model}>
                <Box width={4}>
                  {state.status === "done" ? (
                    <Text color="green"> + </Text>
                  ) : state.status === "error" ? (
                    <Text color="red"> x </Text>
                  ) : state.status === "running" ? (
                    <Box>
                      <Text> </Text>
                      <Spinner />
                      <Text> </Text>
                    </Box>
                  ) : (
                    <Text dimColor> - </Text>
                  )}
                </Box>
                <Box width={30}>
                  <Text bold={state.status === "running"}>
                    {config.type === "voice" ? "Voice Simulation" : model}
                  </Text>
                </Box>
                {state.status === "done" ? (
                  <Text color="green">Complete</Text>
                ) : state.status === "running" ? (
                  <Text color="cyan">Running...</Text>
                ) : state.status === "error" ? (
                  <Text color="red">Failed</Text>
                ) : (
                  <Text dimColor>Waiting</Text>
                )}
              </Box>
            );
          })}

          {/* Log windows for running models */}
          {phase === "eval" && runningModels.length > 0 && (
            <Box flexDirection="row" marginTop={1}>
              {runningModels.map((model, idx) => (
                <Box
                  key={model}
                  flexDirection="column"
                  width={config.type === "voice" ? "100%" : "50%"}
                  marginRight={idx < runningModels.length - 1 ? 1 : 0}
                >
                  <Box>
                    <Text dimColor>{"── "}</Text>
                    <Text bold color="cyan">
                      {config.type === "voice"
                        ? "Voice Simulation"
                        : model.length > 20
                        ? model.slice(-20)
                        : model}
                    </Text>
                    <Text dimColor>
                      {" " +
                        "\u2500".repeat(
                          Math.max(
                            0,
                            20 -
                              Math.min(
                                config.type === "voice" ? 16 : model.length,
                                20
                              )
                          )
                        )}
                    </Text>
                  </Box>
                  <Box flexDirection="column" paddingLeft={1}>
                    {(modelStates[model]?.logs || [])
                      .slice(-8)
                      .map((line, i) => (
                        <Text key={i} dimColor wrap="truncate">
                          {stripAnsi(line).slice(0, 70)}
                        </Text>
                      ))}
                  </Box>
                </Box>
              ))}
            </Box>
          )}

          {phase === "leaderboard" && (
            <Box marginTop={1}>
              <Spinner label="Generating leaderboard..." />
            </Box>
          )}

          {phase === "done" && (
            <Box marginTop={1}>
              <Text color="green">+ All simulations complete!</Text>
            </Box>
          )}
        </Box>
      );
    }

    case "leaderboard": {
      // Handle init error case
      if (initError) {
        return (
          <Box flexDirection="column" padding={1}>
            {header}
            <Text color="red">{initError}</Text>
            <Box marginTop={1}>
              <Text dimColor>Press q to exit</Text>
            </Box>
          </Box>
        );
      }

      const resolvedOutputDir = path.resolve(config.outputDir);

      // Model Detail View
      if (view === "model-detail" && selectedModel) {
        const visibleRows = modelResults.slice(
          scrollOffset,
          scrollOffset + MAX_VISIBLE_ROWS
        );
        const truncate = (s: string, max: number) =>
          s.length > max ? s.slice(0, max - 1) + "…" : s;

        // Get metric columns dynamically (excluding persona_idx, scenario_idx)
        const metricColumns =
          modelResults.length > 0
            ? Object.keys(modelResults[0]!).filter(
                (k) =>
                  !["persona_idx", "scenario_idx"].includes(k) &&
                  typeof modelResults[0]![k] === "number"
              )
            : [];

        return (
          <Box flexDirection="column" padding={1}>
            <Box marginBottom={1}>
              <Text bold color="cyan">
                {config.type === "voice" ? "Voice Simulation" : selectedModel} —
                Results
              </Text>
              <Text dimColor> ({modelResults.length} simulations)</Text>
            </Box>

            {modelResults.length === 0 ? (
              <Text color="yellow">No results found.</Text>
            ) : (
              <>
                {/* Results Table */}
                <Table
                  columns={[
                    { key: "persona", label: "Persona", width: 8 },
                    { key: "scenario", label: "Scenario", width: 10 },
                    ...metricColumns.slice(0, 4).map((col) => ({
                      key: col,
                      label: truncate(col.replace(/_/g, " "), 12),
                      width: 12,
                      align: "right" as const,
                    })),
                  ]}
                  data={visibleRows.map((r) => {
                    const row: Record<string, string> = {
                      persona: String(r.persona_idx ?? "-"),
                      scenario: String(r.scenario_idx ?? "-"),
                    };
                    for (const col of metricColumns.slice(0, 4)) {
                      const val = r[col];
                      row[col] =
                        typeof val === "number"
                          ? val.toFixed(2)
                          : String(val ?? "-");
                    }
                    return row;
                  })}
                />

                {/* Scroll indicator */}
                {modelResults.length > MAX_VISIBLE_ROWS && (
                  <Box marginTop={1}>
                    <Text dimColor>
                      Showing {scrollOffset + 1}-
                      {Math.min(
                        scrollOffset + MAX_VISIBLE_ROWS,
                        modelResults.length
                      )}{" "}
                      of {modelResults.length} | Use ↑↓ to scroll
                    </Text>
                  </Box>
                )}
              </>
            )}

            <Box marginTop={1}>
              <Text dimColor>Press q or Esc to go back to leaderboard</Text>
            </Box>
          </Box>
        );
      }

      // Leaderboard View (default)
      if (metrics.length === 0) {
        return (
          <Box padding={1} flexDirection="column">
            <Text color="green" bold>
              Simulation complete!
            </Text>
            <Box marginTop={1} flexDirection="column">
              <Text bold>Output:</Text>
              <Box>
                <Text>{"  Results: "}</Text>
                <Text color="cyan">{resolvedOutputDir}</Text>
              </Box>
            </Box>
            <Box marginTop={1}>
              <Text dimColor>Press q to exit</Text>
            </Box>
          </Box>
        );
      }

      // Get metric keys for charts (exclude 'model' and 'overall')
      const metricKeys =
        metrics.length > 0
          ? Object.keys(metrics[0]!).filter(
              (k) => k !== "model" && typeof metrics[0]![k] === "number"
            )
          : [];

      // Build table columns dynamically
      const tableColumns = [
        { key: "model", label: "Model", width: 28 },
        ...metricKeys.slice(0, 4).map((k) => ({
          key: k,
          label: k.replace(/_/g, " ").slice(0, 12),
          width: 12,
          align: "right" as const,
        })),
      ];

      return (
        <Box flexDirection="column" padding={1}>
          <Box marginBottom={1}>
            <Text bold color="cyan">
              {config.type === "text"
                ? "Simulations Leaderboard"
                : "Simulation Results"}
            </Text>
          </Box>

          {/* Comparison Table */}
          <Table
            columns={tableColumns}
            data={metrics.map((m) => {
              const row: Record<string, string> = { model: String(m.model) };
              for (const k of metricKeys.slice(0, 4)) {
                const val = m[k];
                row[k] =
                  typeof val === "number" ? val.toFixed(2) : String(val ?? "-");
              }
              return row;
            })}
          />

          {/* Charts for top metrics */}
          {metricKeys.slice(0, 2).map((metricKey) => (
            <Box key={metricKey} marginTop={1} flexDirection="column">
              <Text bold>{metricKey.replace(/_/g, " ")}</Text>
              <BarChart
                data={[...metrics]
                  .sort(
                    (a, b) =>
                      ((b[metricKey] as number) || 0) -
                      ((a[metricKey] as number) || 0)
                  )
                  .map((m) => ({
                    label:
                      String(m.model).length > 25
                        ? String(m.model).slice(-25)
                        : String(m.model),
                    value: (m[metricKey] as number) || 0,
                    color: "green",
                  }))}
                maxWidth={40}
              />
            </Box>
          ))}

          {/* Model selection to view details (for text simulations with multiple models) */}
          {config.type === "text" && config.models.length > 0 && (
            <Box marginTop={1} flexDirection="column">
              <Text dimColor>{"\u2500".repeat(50)}</Text>
              <Box marginTop={1}>
                <Text bold>View Model Details</Text>
              </Box>
              <Box marginTop={1}>
                <SelectInput
                  items={[
                    ...config.models.map((m) => ({
                      label: `${m} — View simulation results`,
                      value: m,
                    })),
                    { label: "Exit", value: "__exit__" },
                  ]}
                  onSelect={(v) => {
                    if (v === "__exit__") {
                      if (onBack) onBack();
                      else exit();
                    } else {
                      setSelectedModel(v);
                      setView("model-detail");
                    }
                  }}
                />
              </Box>
            </Box>
          )}

          {/* Voice simulation detail view option */}
          {config.type === "voice" && (
            <Box marginTop={1} flexDirection="column">
              <Text dimColor>{"\u2500".repeat(50)}</Text>
              <Box marginTop={1}>
                <SelectInput
                  items={[
                    { label: "View simulation details", value: "voice" },
                    { label: "Exit", value: "__exit__" },
                  ]}
                  onSelect={(v) => {
                    if (v === "__exit__") {
                      if (onBack) onBack();
                      else exit();
                    } else {
                      setSelectedModel("voice");
                      setView("model-detail");
                    }
                  }}
                />
              </Box>
            </Box>
          )}

          {/* Output file paths */}
          <Box marginTop={1} flexDirection="column">
            <Text dimColor>{"\u2500".repeat(50)}</Text>
            <Box marginTop={1} flexDirection="column">
              <Text bold>Output Files</Text>
              <Box>
                <Text>{"  Results: "}</Text>
                <Text color="cyan">{resolvedOutputDir}</Text>
              </Box>
            </Box>
          </Box>
        </Box>
      );
    }

    default:
      return null;
  }
}
