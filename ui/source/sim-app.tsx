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

interface TranscriptMessage {
  role: string;
  content?: string;
  tool_calls?: Array<{
    id: string;
    function: { name: string; arguments: string };
    type: string;
  }>;
  tool_call_id?: string;
}

interface PersonaInfo {
  label?: string;
  characteristics?: string;
  gender?: string;
  language?: string;
}

interface ScenarioInfo {
  name?: string;
  description?: string;
}

interface EvalResult {
  simulation: string;
  persona_idx: number;
  scenario_idx: number;
  criteria: { name: string; value: number; reasoning: string }[];
  transcript: TranscriptMessage[];
  personaInfo?: PersonaInfo;
  scenarioInfo?: ScenarioInfo;
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

  // ── config data for personas/scenarios ──
  const [personas, setPersonas] = useState<PersonaInfo[]>([]);
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);

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
      const selectedResult = evalResults.find(
        (r) => r.simulation === selectedSim
      );
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
        if (a.personaIdx !== b.personaIdx) return a.personaIdx - b.personaIdx;
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

  // ── Load metrics from metrics.json ──
  const loadMetrics = () => {
    try {
      const metricsPath = path.join(config.outputDir, "metrics.json");
      if (fs.existsSync(metricsPath)) {
        const raw = JSON.parse(fs.readFileSync(metricsPath, "utf-8"));
        const parsed: typeof metrics = {};
        for (const [key, val] of Object.entries(raw)) {
          if (
            typeof val === "object" &&
            val !== null &&
            "mean" in val &&
            "std" in val &&
            "values" in val
          ) {
            parsed[key] = val as {
              mean: number;
              std: number;
              values: number[];
            };
          }
        }
        setMetrics(parsed);
      }
    } catch {
      // Ignore
    }
  };

  // ── Load personas and scenarios from config file ──
  const loadConfigData = () => {
    try {
      if (!config.configPath || !fs.existsSync(config.configPath)) return;
      const configData = JSON.parse(
        fs.readFileSync(config.configPath, "utf-8")
      );
      if (configData.personas && Array.isArray(configData.personas)) {
        setPersonas(configData.personas);
      }
      if (configData.scenarios && Array.isArray(configData.scenarios)) {
        setScenarios(configData.scenarios);
      }
    } catch {
      // Ignore parse errors
    }
  };

  // ── Load evaluation results from each simulation directory ──
  const loadEvalResults = () => {
    // First load config data to get personas/scenarios
    loadConfigData();

    const results: EvalResult[] = [];

    try {
      if (!fs.existsSync(config.outputDir)) return;

      const entries = fs.readdirSync(config.outputDir, { withFileTypes: true });
      const simDirs = entries
        .filter(
          (e) => e.isDirectory() && e.name.startsWith("simulation_persona_")
        )
        .map((e) => e.name);

      // Load personas/scenarios from config
      let loadedPersonas: PersonaInfo[] = [];
      let loadedScenarios: ScenarioInfo[] = [];
      try {
        if (config.configPath && fs.existsSync(config.configPath)) {
          const configData = JSON.parse(
            fs.readFileSync(config.configPath, "utf-8")
          );
          loadedPersonas = configData.personas || [];
          loadedScenarios = configData.scenarios || [];
        }
      } catch {
        // Ignore
      }

      for (const dirName of simDirs) {
        const match = dirName.match(/simulation_persona_(\d+)_scenario_(\d+)/);
        if (!match) continue;

        const personaIdx = parseInt(match[1]!, 10);
        const scenarioIdx = parseInt(match[2]!, 10);

        // Get persona and scenario info (1-indexed in folder name)
        const personaInfo = loadedPersonas[personaIdx - 1];
        const scenarioInfo = loadedScenarios[scenarioIdx - 1];

        // Read evaluation_results.csv
        const evalPath = path.join(
          config.outputDir,
          dirName,
          "evaluation_results.csv"
        );
        if (!fs.existsSync(evalPath)) continue;

        // Read transcript.json
        let transcript: TranscriptMessage[] = [];
        const transcriptPath = path.join(
          config.outputDir,
          dirName,
          "transcript.json"
        );
        if (fs.existsSync(transcriptPath)) {
          try {
            transcript = JSON.parse(fs.readFileSync(transcriptPath, "utf-8"));
          } catch {
            // Ignore parse errors
          }
        }

        try {
          const content = fs.readFileSync(evalPath, "utf-8");
          const lines = content.trim().split("\n");
          if (lines.length < 2) continue;

          // Parse CSV (name,value,reasoning) - handle quoted fields
          const criteria: { name: string; value: number; reasoning: string }[] =
            [];
          for (let i = 1; i < lines.length; i++) {
            const line = lines[i]!;
            // Handle CSV with potential commas and quotes in reasoning
            const firstComma = line.indexOf(",");
            if (firstComma === -1) continue;

            const name = line.slice(0, firstComma).trim();
            const rest = line.slice(firstComma + 1);

            // Find value (number before next comma or quote)
            let valueStr = "";
            let reasoningStart = 0;
            for (let j = 0; j < rest.length; j++) {
              if (rest[j] === "," || rest[j] === '"') {
                valueStr = rest.slice(0, j).trim();
                reasoningStart = rest[j] === "," ? j + 1 : j;
                break;
              }
            }

            let reasoning = rest.slice(reasoningStart).trim();
            // Remove surrounding quotes if present
            if (reasoning.startsWith('"') && reasoning.endsWith('"')) {
              reasoning = reasoning.slice(1, -1).replace(/""/g, '"');
            }

            const value = parseFloat(valueStr);

            if (!isNaN(value)) {
              criteria.push({ name, value, reasoning });
            }
          }

          results.push({
            simulation: dirName,
            persona_idx: personaIdx,
            scenario_idx: scenarioIdx,
            criteria,
            transcript,
            personaInfo,
            scenarioInfo,
          });
        } catch {
          // Skip files with read errors
        }
      }

      // Sort by persona then scenario
      results.sort((a, b) => {
        if (a.persona_idx !== b.persona_idx)
          return a.persona_idx - b.persona_idx;
        return a.scenario_idx - b.scenario_idx;
      });

      setEvalResults(results);
    } catch {
      // Ignore
    }
  };

  // ── Cleanup on unmount ──
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
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
      // For text simulations: show simulation slots
      // For voice simulations: show single voice state
      const isTextSim = config.type === "text";

      // Get running simulation slots for display (limit to parallel count)
      const runningSlots = simSlots
        .filter((s) => s.status === "running")
        .slice(0, config.parallel);
      const completedSlots = simSlots.filter((s) => s.status === "done");

      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Box marginBottom={1}>
            <Text dimColor>
              Type: {config.type} | Config: {config.configPath}
            </Text>
          </Box>

          {isTextSim ? (
            <>
              {/* Text simulation status */}
              <Box marginBottom={1}>
                <Text dimColor>
                  Model: {config.models[0] || "gpt-4.1"} | Provider:{" "}
                  {config.provider}
                  {config.parallel > 1 && ` | Parallel: ${config.parallel}`}
                </Text>
              </Box>
              <Box marginBottom={1}>
                <Text>
                  {simProcessRunning ? (
                    <>
                      <Spinner /> Running simulations...
                    </>
                  ) : (
                    <Text color="green">+ Complete</Text>
                  )}
                </Text>
                {simSlots.length > 0 && (
                  <Text dimColor>
                    {" "}
                    ({completedSlots.length}/{simSlots.length} done)
                  </Text>
                )}
              </Box>

              {/* Simulation slot status list */}
              {simSlots.map((slot) => (
                <Box key={slot.name}>
                  <Box width={4}>
                    {slot.status === "done" ? (
                      <Text color="green"> + </Text>
                    ) : (
                      <Box>
                        <Text> </Text>
                        <Spinner />
                        <Text> </Text>
                      </Box>
                    )}
                  </Box>
                  <Box width={35}>
                    <Text bold={slot.status === "running"}>
                      Persona {slot.personaIdx} Scenario {slot.scenarioIdx}
                    </Text>
                  </Box>
                  {slot.status === "done" ? (
                    <Text color="green">Complete</Text>
                  ) : (
                    <Text color="cyan">Running...</Text>
                  )}
                </Box>
              ))}

              {/* Log windows for running slots - side by side */}
              {phase === "eval" && runningSlots.length > 0 && (
                <Box flexDirection="row" marginTop={1}>
                  {runningSlots.map((slot, idx) => (
                    <Box
                      key={slot.name}
                      flexDirection="column"
                      width={
                        runningSlots.length === 1
                          ? "100%"
                          : `${Math.floor(100 / runningSlots.length)}%`
                      }
                      marginRight={idx < runningSlots.length - 1 ? 1 : 0}
                    >
                      <Box>
                        <Text dimColor>{"── "}</Text>
                        <Text bold color="cyan">
                          P{slot.personaIdx} S{slot.scenarioIdx}
                        </Text>
                        <Text dimColor>{" " + "\u2500".repeat(15)}</Text>
                      </Box>
                      <Box flexDirection="column" paddingLeft={1}>
                        {slot.logs.slice(-8).map((line, i) => (
                          <Text key={i} dimColor wrap="truncate">
                            {stripAnsi(line).slice(0, 50)}
                          </Text>
                        ))}
                      </Box>
                    </Box>
                  ))}
                </Box>
              )}
            </>
          ) : (
            <>
              {/* Voice simulation status */}
              {modelStates.voice && (
                <>
                  <Box>
                    <Box width={4}>
                      {modelStates.voice.status === "done" ? (
                        <Text color="green"> + </Text>
                      ) : modelStates.voice.status === "error" ? (
                        <Text color="red"> x </Text>
                      ) : (
                        <Box>
                          <Text> </Text>
                          <Spinner />
                          <Text> </Text>
                        </Box>
                      )}
                    </Box>
                    <Box width={30}>
                      <Text bold={modelStates.voice.status === "running"}>
                        Voice Simulation
                      </Text>
                    </Box>
                    {modelStates.voice.status === "done" ? (
                      <Text color="green">Complete</Text>
                    ) : modelStates.voice.status === "error" ? (
                      <Text color="red">Failed</Text>
                    ) : (
                      <Text color="cyan">Running...</Text>
                    )}
                  </Box>

                  {/* Voice simulation logs */}
                  {phase === "eval" &&
                    modelStates.voice.status === "running" && (
                      <Box flexDirection="column" marginTop={1}>
                        <Box>
                          <Text dimColor>{"── "}</Text>
                          <Text bold color="cyan">
                            Voice Simulation
                          </Text>
                          <Text dimColor>{" " + "\u2500".repeat(15)}</Text>
                        </Box>
                        <Box flexDirection="column" paddingLeft={1}>
                          {modelStates.voice.logs.slice(-8).map((line, i) => (
                            <Text key={i} dimColor wrap="truncate">
                              {stripAnsi(line).slice(0, 70)}
                            </Text>
                          ))}
                        </Box>
                      </Box>
                    )}
                </>
              )}
            </>
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
      const metricKeys = Object.keys(metrics);
      const truncate = (s: string, max: number) =>
        s.length > max ? s.slice(0, max - 1) + "…" : s;

      // Simulation Detail View - show transcript and evaluation
      if (view === "sim-detail" && selectedSim) {
        const selectedResult = evalResults.find(
          (r) => r.simulation === selectedSim
        );
        const criteria = selectedResult?.criteria || [];
        const transcript = selectedResult?.transcript || [];

        // Format tool calls for display
        const formatToolCall = (
          tc: TranscriptMessage["tool_calls"]
        ): string => {
          if (!tc || tc.length === 0) return "";
          return tc
            .map((t) => {
              try {
                const args = JSON.parse(t.function.arguments);
                return `${t.function.name}(${JSON.stringify(args)})`;
              } catch {
                return `${t.function.name}(${t.function.arguments})`;
              }
            })
            .join(", ");
        };

        return (
          <Box flexDirection="column" padding={1}>
            {/* Header with persona/scenario info */}
            <Box marginBottom={1} flexDirection="column">
              <Text bold color="cyan">
                Simulation Details
              </Text>
              {selectedResult?.personaInfo && (
                <Box marginTop={1}>
                  <Text bold>Persona: </Text>
                  <Text>
                    {selectedResult.personaInfo.label ||
                      `Persona ${selectedResult.persona_idx}`}
                  </Text>
                </Box>
              )}
              {selectedResult?.scenarioInfo && (
                <Box>
                  <Text bold>Scenario: </Text>
                  <Text>
                    {selectedResult.scenarioInfo.name ||
                      `Scenario ${selectedResult.scenario_idx}`}
                  </Text>
                </Box>
              )}
            </Box>

            {/* Transcript Section - show all messages */}
            <Box marginBottom={1}>
              <Text bold>Transcript</Text>
            </Box>
            <Box
              flexDirection="column"
              marginBottom={1}
              borderStyle="single"
              borderColor="gray"
              paddingX={1}
            >
              {transcript.length === 0 ? (
                <Text dimColor>No transcript available</Text>
              ) : (
                transcript
                  .filter((m) => m.role !== "tool") // Skip tool responses for cleaner view
                  .map((m, idx) => (
                    <Box key={idx} flexDirection="column" marginBottom={1}>
                      <Text
                        color={
                          m.role === "assistant"
                            ? "cyan"
                            : m.role === "user"
                            ? "yellow"
                            : "gray"
                        }
                        bold
                      >
                        {m.role}:
                      </Text>
                      <Text wrap="wrap">
                        {m.content ||
                          (m.tool_calls ? formatToolCall(m.tool_calls) : "")}
                      </Text>
                    </Box>
                  ))
              )}
            </Box>

            {/* Evaluation Metrics Section */}
            <Box marginBottom={1}>
              <Text bold>Evaluation</Text>
            </Box>
            {criteria.length === 0 ? (
              <Text color="yellow">No evaluation results found.</Text>
            ) : (
              criteria.map((c, idx) => (
                <Box
                  key={idx}
                  flexDirection="column"
                  marginBottom={1}
                  borderStyle="single"
                  borderColor={c.value >= 0.5 ? "green" : "red"}
                  paddingX={1}
                >
                  <Box>
                    <Text bold>{c.name.replace(/_/g, " ")}</Text>
                    <Text> </Text>
                    {c.value >= 0.5 ? (
                      <Text color="green" bold>
                        {c.value.toFixed(1)} ✓
                      </Text>
                    ) : (
                      <Text color="red" bold>
                        {c.value.toFixed(1)} ✗
                      </Text>
                    )}
                  </Box>
                  <Box marginTop={1}>
                    <Text wrap="wrap">{c.reasoning}</Text>
                  </Box>
                </Box>
              ))
            )}

            <Box marginTop={1}>
              <Text dimColor>Press q or Esc to go back to results</Text>
            </Box>
          </Box>
        );
      }

      // Main Results View (default)
      const hasMetrics = metricKeys.length > 0;
      const hasEvalResults = evalResults.length > 0;

      if (!hasMetrics && !hasEvalResults) {
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

      return (
        <Box flexDirection="column" padding={1}>
          <Box marginBottom={1}>
            <Text bold color="cyan">
              Simulation Results
            </Text>
            <Text dimColor>
              {" "}
              — {evalResults.length} simulation
              {evalResults.length !== 1 ? "s" : ""}
            </Text>
          </Box>

          {/* Overall Metrics Bar Charts - simplified */}
          {hasMetrics && (
            <>
              <Box marginBottom={1}>
                <Text bold>Overall Metrics</Text>
              </Box>
              {metricKeys.map((metricKey) => {
                const m = metrics[metricKey]!;
                return (
                  <Box key={metricKey} marginBottom={1}>
                    <BarChart
                      data={[
                        {
                          label: metricKey.replace(/_/g, " "),
                          value: m.mean,
                          color: m.mean >= 0.5 ? "green" : "red",
                        },
                      ]}
                      maxWidth={40}
                    />
                  </Box>
                );
              })}
            </>
          )}

          {/* Per-Simulation Results - Cards with SelectInput */}
          {hasEvalResults && (
            <>
              {/* Cards displayed above */}
              {evalResults.map((r, idx) => {
                const personaLabel =
                  r.personaInfo?.label || `Persona ${r.persona_idx}`;
                const scenarioLabel =
                  r.scenarioInfo?.name || `Scenario ${r.scenario_idx}`;

                return (
                  <Box
                    key={idx}
                    flexDirection="column"
                    marginBottom={1}
                    borderStyle="single"
                    borderColor="gray"
                    paddingX={1}
                  >
                    <Box>
                      <Text bold>Persona: </Text>
                      <Text>{personaLabel}</Text>
                    </Box>
                    <Box>
                      <Text bold>Scenario: </Text>
                      <Text>{scenarioLabel}</Text>
                    </Box>
                    {r.criteria.length > 0 && (
                      <Box flexDirection="column" marginTop={1}>
                        {r.criteria.map((c, cIdx) => (
                          <Box key={cIdx}>
                            <Box width={30}>
                              <Text dimColor>{c.name.replace(/_/g, " ")}</Text>
                            </Box>
                            <Text color={c.value >= 0.5 ? "green" : "red"}>
                              {c.value.toFixed(1)} {c.value >= 0.5 ? "✓" : "✗"}
                            </Text>
                          </Box>
                        ))}
                      </Box>
                    )}
                  </Box>
                );
              })}

              {/* Select to view details */}
              <Box marginTop={1}>
                <Text dimColor>Select to view transcript & evaluation:</Text>
              </Box>
              <SelectInput
                items={[
                  ...evalResults.map((r) => {
                    const personaLabel =
                      r.personaInfo?.label || `Persona ${r.persona_idx}`;
                    const scenarioLabel =
                      r.scenarioInfo?.name || `Scenario ${r.scenario_idx}`;
                    return {
                      label: `${personaLabel} — ${scenarioLabel}`,
                      value: r.simulation,
                    };
                  }),
                  { label: "Exit", value: "__exit__" },
                ]}
                onSelect={(v) => {
                  if (v === "__exit__") {
                    if (onBack) onBack();
                    else exit();
                  } else {
                    setSelectedSim(v);
                    setScrollOffset(0);
                    setView("sim-detail");
                  }
                }}
              />
            </>
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
