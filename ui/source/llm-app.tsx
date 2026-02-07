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
  MultiSelect,
} from "./components.js";
import { getCredential, saveCredential } from "./credentials.js";
import { type CalibrateCmd, findCalibrateBin, stripAnsi } from "./shared.js";

// ─── Types ───────────────────────────────────────────────────
type LlmStep =
  | "init"
  | "config-path"
  | "provider"
  | "select-models"
  | "output-dir"
  | "output-dir-confirm"
  | "api-keys"
  | "running"
  | "leaderboard";

interface ModelState {
  status: "waiting" | "running" | "done" | "error";
  logs: string[];
  metrics?: { passed?: number; failed?: number; total?: number };
}

interface TestResult {
  id: string;
  input?: string;
  expected_output?: string;
  actual_output?: string;
  passed?: boolean;
  reason?: string;
  [key: string]: string | number | boolean | undefined;
}

const MAX_PARALLEL_MODELS = 2;

// ─── Model Options ───────────────────────────────────────────
const OPENAI_MODELS = [
  { label: "gpt-4.1", value: "gpt-4.1" },
  { label: "gpt-4.1-mini", value: "gpt-4.1-mini" },
  { label: "gpt-4o", value: "gpt-4o" },
  { label: "gpt-4o-mini", value: "gpt-4o-mini" },
  { label: "o1", value: "o1" },
  { label: "o1-mini", value: "o1-mini" },
  { label: "o3-mini", value: "o3-mini" },
];

const OPENROUTER_MODELS = [
  { label: "openai/gpt-4.1", value: "openai/gpt-4.1" },
  { label: "openai/gpt-4.1-mini", value: "openai/gpt-4.1-mini" },
  { label: "openai/gpt-4o", value: "openai/gpt-4o" },
  { label: "anthropic/claude-sonnet-4", value: "anthropic/claude-sonnet-4" },
  {
    label: "anthropic/claude-3.5-sonnet",
    value: "anthropic/claude-3.5-sonnet",
  },
  {
    label: "google/gemini-2.0-flash-001",
    value: "google/gemini-2.0-flash-001",
  },
  {
    label: "google/gemini-2.5-pro-preview",
    value: "google/gemini-2.5-pro-preview",
  },
];

interface LlmConfig {
  configPath: string;
  models: string[];
  provider: string;
  outputDir: string;
  overwrite: boolean;
  envVars: Record<string, string>;
  calibrate: CalibrateCmd;
}

// ─── Main Component ──────────────────────────────────────────
export function LlmTestsApp({ onBack }: { onBack?: () => void }) {
  const { exit } = useApp();
  const [step, setStep] = useState<LlmStep>("init");
  const [config, setConfig] = useState<LlmConfig>({
    configPath: "",
    models: [],
    provider: "openrouter",
    outputDir: "./out",
    overwrite: false,
    envVars: {},
    calibrate: { cmd: "calibrate", args: [] },
  });

  // ── overwrite confirmation state ──
  const [existingDirs, setExistingDirs] = useState<string[]>([]);

  // ── input state ──
  const [configInput, setConfigInput] = useState("");
  const [outputInput, setOutputInput] = useState("./out");

  // ── API key state ──
  const [missingKeys, setMissingKeys] = useState<string[]>([]);
  const [currentKeyIdx, setCurrentKeyIdx] = useState(0);
  const [keyInput, setKeyInput] = useState("");

  // ── run state ──
  const [modelStates, setModelStates] = useState<Record<string, ModelState>>(
    {}
  );
  const [phase, setPhase] = useState<"eval" | "leaderboard" | "done">("eval");
  const [runningCount, setRunningCount] = useState(0);
  const [nextModelIdx, setNextModelIdx] = useState(0);
  const processRefs = useRef<Map<string, ChildProcess>>(new Map());
  const calibrateBin = useRef<CalibrateCmd | null>(null);

  // ── init error state ──
  const [initError, setInitError] = useState("");

  // ── leaderboard state ──
  const [view, setView] = useState<"leaderboard" | "model-detail">(
    "leaderboard"
  );
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [modelResults, setModelResults] = useState<TestResult[]>([]);
  const [scrollOffset, setScrollOffset] = useState(0);
  const MAX_VISIBLE_ROWS = 10;

  const [metrics, setMetrics] = useState<
    Array<{
      model: string;
      passed: number;
      failed: number;
      total: number;
      pass_rate: number;
      overall?: number;
    }>
  >([]);

  // Step navigation helper
  const goBack = () => {
    switch (step) {
      case "config-path":
        if (onBack) onBack();
        break;
      case "provider":
        setStep("config-path");
        break;
      case "select-models":
        setStep("provider");
        break;
      case "output-dir":
        setStep("select-models");
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
        if (view === "model-detail") {
          setView("leaderboard");
          setSelectedModel(null);
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
      if (step === "leaderboard" && view === "model-detail") {
        setView("leaderboard");
        setSelectedModel(null);
        setScrollOffset(0);
      } else if (!["init", "running", "leaderboard"].includes(step)) {
        goBack();
      }
    }
    // Scroll in model detail view
    if (step === "leaderboard" && view === "model-detail") {
      if (key.upArrow && scrollOffset > 0) {
        setScrollOffset((o) => o - 1);
      }
      if (
        key.downArrow &&
        scrollOffset < modelResults.length - MAX_VISIBLE_ROWS
      ) {
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
    setStep("config-path");
  }, [step]);

  // ── Check API keys ──
  function checkApiKeys(provider: string) {
    const needed: string[] = [];
    if (provider === "openrouter") {
      if (!getCredential("OPENROUTER_API_KEY"))
        needed.push("OPENROUTER_API_KEY");
    }
    if (!getCredential("OPENAI_API_KEY")) needed.push("OPENAI_API_KEY");
    return needed;
  }

  // ── Build model directory name (matches Python logic) ──
  function getModelDir(model: string): string {
    let modelDir =
      config.provider === "openai" ? `${config.provider}/${model}` : model;
    return modelDir.replace(/\//g, "__");
  }

  // ── Initialize model states when entering running step ──
  useEffect(() => {
    if (step !== "running") return;

    // Initialize model states
    const initialStates: Record<string, ModelState> = {};
    for (const model of config.models) {
      initialStates[model] = { status: "waiting", logs: [] };
    }
    setModelStates(initialStates);
    setPhase("eval");
    setRunningCount(0);
    setNextModelIdx(0);
  }, [step]);

  // ── Start a single model evaluation ──
  const startModel = (model: string) => {
    if (!config.calibrate) return;

    const bin = config.calibrate;
    const env: Record<string, string> = { ...process.env } as Record<
      string,
      string
    >;

    // Inject stored credentials and config env vars
    for (const k of ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]) {
      const v = getCredential(k);
      if (v) env[k] = v;
    }
    Object.assign(env, config.envVars);
    env.PYTHONUNBUFFERED = "1";

    const cmdArgs = [
      ...bin.args,
      "llm",
      "-c",
      config.configPath,
      "-o",
      config.outputDir,
      "-m",
      model,
      "-p",
      config.provider,
    ];

    setModelStates((prev) => ({
      ...prev,
      [model]: { ...prev[model]!, status: "running" },
    }));
    setRunningCount((c) => c + 1);

    const proc = spawn(bin.cmd, cmdArgs, {
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });

    processRefs.current.set(model, proc);

    const onData = (data: Buffer) => {
      const lines = data
        .toString()
        .split(/[\r\n]+/)
        .filter((l) => l.trim());
      setModelStates((prev) => ({
        ...prev,
        [model]: {
          ...prev[model]!,
          logs: [...prev[model]!.logs, ...lines].slice(-20),
        },
      }));
    };

    proc.stdout?.on("data", onData);
    proc.stderr?.on("data", onData);

    proc.on("error", () => {
      setModelStates((prev) => ({
        ...prev,
        [model]: { ...prev[model]!, status: "error" },
      }));
      setRunningCount((c) => c - 1);
      processRefs.current.delete(model);
    });

    proc.on("close", (code) => {
      // Try to read metrics from results.json
      let metricsData: ModelState["metrics"] = undefined;
      if (code === 0) {
        try {
          const modelDir = getModelDir(model);
          const resultsPath = path.join(
            config.outputDir,
            modelDir,
            "results.json"
          );
          if (fs.existsSync(resultsPath)) {
            const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
            const passed = results.filter(
              (r: { metrics?: { passed?: boolean } }) => r.metrics?.passed
            ).length;
            const total = results.length;
            metricsData = { passed, failed: total - passed, total };
          }
        } catch {
          // Ignore errors reading metrics
        }
      }

      setModelStates((prev) => ({
        ...prev,
        [model]: {
          ...prev[model]!,
          status: code === 0 ? "done" : "error",
          metrics: metricsData,
        },
      }));
      setRunningCount((c) => c - 1);
      processRefs.current.delete(model);
    });
  };

  // ── Effect to manage parallel model execution ──
  useEffect(() => {
    if (step !== "running" || phase !== "eval") return;
    if (Object.keys(modelStates).length === 0) return;

    // Check if all models are done
    const completedCount = Object.values(modelStates).filter(
      (s) => s.status === "done" || s.status === "error"
    ).length;

    if (completedCount >= config.models.length) {
      // All models done, generate leaderboard then finish
      setPhase("leaderboard");

      const env: Record<string, string> = { ...process.env } as Record<
        string,
        string
      >;
      env.PYTHONUNBUFFERED = "1";

      const lbDir = path.join(config.outputDir, "leaderboard");

      // Generate leaderboard using python -m calibrate.llm.tests_leaderboard
      const proc = spawn(
        "python",
        [
          "-m",
          "calibrate.llm.tests_leaderboard",
          "-o",
          config.outputDir,
          "-s",
          lbDir,
        ],
        { env, stdio: ["pipe", "pipe", "pipe"] }
      );

      proc.on("close", () => {
        loadMetrics();
        setPhase("done");
        setTimeout(() => setStep("leaderboard"), 500);
      });

      proc.on("error", () => {
        loadMetrics();
        setPhase("done");
        setTimeout(() => setStep("leaderboard"), 500);
      });
      return;
    }

    // Start more models if we have capacity
    if (
      runningCount < MAX_PARALLEL_MODELS &&
      nextModelIdx < config.models.length
    ) {
      const model = config.models[nextModelIdx]!;
      setNextModelIdx((idx) => idx + 1);
      startModel(model);
    }
  }, [step, phase, runningCount, nextModelIdx, modelStates]);

  // ── Load metrics for leaderboard ──
  const loadMetrics = () => {
    const results: typeof metrics = [];
    for (const model of config.models) {
      try {
        const modelDir = getModelDir(model);
        const resultsPath = path.join(
          config.outputDir,
          modelDir,
          "results.json"
        );
        if (fs.existsSync(resultsPath)) {
          const data = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
          const passed = data.filter(
            (r: { metrics?: { passed?: boolean } }) => r.metrics?.passed
          ).length;
          const total = data.length;
          const failed = total - passed;
          const pass_rate = total > 0 ? (passed / total) * 100 : 0;
          results.push({ model, passed, failed, total, pass_rate });
        }
      } catch {
        // Skip models with no results
      }
    }

    // Try to read overall scores from leaderboard CSV
    try {
      const lbCsvPath = path.join(
        config.outputDir,
        "leaderboard",
        "llm_leaderboard.csv"
      );
      if (fs.existsSync(lbCsvPath)) {
        const lines = fs.readFileSync(lbCsvPath, "utf-8").trim().split("\n");
        if (lines.length > 1) {
          const headers = lines[0]!.split(",").map((h) => h.trim());
          const overallIdx = headers.findIndex(
            (h) => h.toLowerCase() === "overall"
          );
          const modelIdx = headers.findIndex(
            (h) => h.toLowerCase() === "model" || h.toLowerCase() === "provider"
          );

          if (overallIdx >= 0 && modelIdx >= 0) {
            for (let i = 1; i < lines.length; i++) {
              const vals = lines[i]!.split(",");
              const modelName = vals[modelIdx]?.trim() || "";
              const overall = parseFloat(vals[overallIdx] || "0") || 0;
              const result = results.find((r) => r.model === modelName);
              if (result) {
                result.overall = overall;
              }
            }
          }
        }
      }
    } catch {
      // Ignore leaderboard parse errors
    }

    setMetrics(results);
  };

  // ── Load model results when selected ──
  useEffect(() => {
    if (!selectedModel) return;
    try {
      const modelDir = getModelDir(selectedModel);
      const resultsPath = path.join(config.outputDir, modelDir, "results.json");
      if (fs.existsSync(resultsPath)) {
        const data = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
        const results: TestResult[] = data.map(
          (
            r: {
              test_case?: {
                id?: string;
                input?: string;
                expected_output?: string;
              };
              output?: string;
              metrics?: { passed?: boolean; reason?: string };
            },
            idx: number
          ) => ({
            id: r.test_case?.id || String(idx + 1),
            input: r.test_case?.input || "",
            expected_output: r.test_case?.expected_output || "",
            actual_output: r.output || "",
            passed: r.metrics?.passed || false,
            reason: r.metrics?.reason || "",
          })
        );
        setModelResults(results);
        setScrollOffset(0);
      } else {
        setModelResults([]);
      }
    } catch {
      setModelResults([]);
    }
  }, [selectedModel, config.outputDir]);

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
        LLM Tests
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

    case "config-path":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            Path to a JSON config file containing system prompt, tools, and test
            cases.
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
                  setStep("provider");
                }
              }}
              placeholder="./config.json"
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>
              Enter to submit{onBack ? ", Esc to go back" : ""}
            </Text>
          </Box>
        </Box>
      );

    case "provider":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>Which LLM provider to use for running tests.</Text>
          <Text>Provider:</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "OpenRouter", value: "openrouter" },
                { label: "OpenAI", value: "openai" },
              ]}
              onSelect={(v) => {
                setConfig((c) => ({ ...c, provider: v, models: [] }));
                setStep("select-models");
              }}
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>Press Esc to go back</Text>
          </Box>
        </Box>
      );

    case "select-models": {
      const modelOptions =
        config.provider === "openai" ? OPENAI_MODELS : OPENROUTER_MODELS;
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Box marginBottom={1}>
            <Text dimColor>Provider: {config.provider}</Text>
          </Box>
          <Text>Select models to evaluate:</Text>
          <Box marginTop={1}>
            <MultiSelect
              items={modelOptions}
              onSubmit={(selected) => {
                if (selected.length > 0) {
                  setConfig((c) => ({ ...c, models: selected }));
                  setStep("output-dir");
                }
              }}
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>
              Space to toggle, Enter to confirm, Esc to go back
            </Text>
          </Box>
        </Box>
      );
    }

    case "output-dir":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>Directory where test results will be saved.</Text>
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
                const missing = checkApiKeys(config.provider);
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
          <Text>The following directories already contain data:</Text>
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
                  const missing = checkApiKeys(config.provider);
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
      const completedCount = Object.values(modelStates).filter(
        (s) => s.status === "done" || s.status === "error"
      ).length;

      // Get currently running models for log display
      const runningModels = config.models.filter(
        (m) => modelStates[m]?.status === "running"
      );

      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Box marginBottom={1}>
            <Text dimColor>Config: {config.configPath}</Text>
          </Box>
          <Box marginBottom={1}>
            <Text dimColor>
              {completedCount}/{config.models.length} models
              {runningCount > 1 && ` (${runningCount} running in parallel)`}
              {" | "}Provider: {config.provider}
            </Text>
          </Box>

          {/* Model status list */}
          {config.models.map((model) => {
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
                  <Text bold={state.status === "running"}>{model}</Text>
                </Box>
                {state.status === "done" && state.metrics ? (
                  <Text dimColor>
                    {state.metrics.passed}/{state.metrics.total} passed
                  </Text>
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

          {/* Log windows for running models - side by side columns */}
          {phase === "eval" && runningModels.length > 0 && (
            <Box flexDirection="row" marginTop={1}>
              {runningModels.map((model, idx) => (
                <Box
                  key={model}
                  flexDirection="column"
                  width="50%"
                  marginRight={idx < runningModels.length - 1 ? 1 : 0}
                >
                  <Box>
                    <Text dimColor>{"── "}</Text>
                    <Text bold color="cyan">
                      {model.length > 20 ? model.slice(-20) : model}
                    </Text>
                    <Text dimColor>
                      {" " +
                        "\u2500".repeat(
                          Math.max(0, 20 - Math.min(model.length, 20))
                        )}
                    </Text>
                  </Box>
                  <Box flexDirection="column" paddingLeft={1}>
                    {(modelStates[model]?.logs || [])
                      .slice(-8)
                      .map((line, i) => {
                        const cleanLine = stripAnsi(line).slice(0, 45);
                        const isPass =
                          cleanLine.includes("passed") ||
                          cleanLine.includes("✅");
                        const isFail =
                          cleanLine.includes("failed") ||
                          cleanLine.includes("❌");
                        return (
                          <Text
                            key={i}
                            color={
                              isPass ? "green" : isFail ? "red" : undefined
                            }
                            dimColor={!isPass && !isFail}
                            wrap="truncate"
                          >
                            {cleanLine}
                          </Text>
                        );
                      })}
                  </Box>
                </Box>
              ))}
            </Box>
          )}

          {phase === "done" && (
            <Box marginTop={1}>
              <Text color="green">+ All tests complete!</Text>
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

      const leaderboardDir = path.join(config.outputDir, "leaderboard");
      const resolvedOutputDir = path.resolve(config.outputDir);

      // Model Detail View
      if (view === "model-detail" && selectedModel) {
        const visibleRows = modelResults.slice(
          scrollOffset,
          scrollOffset + MAX_VISIBLE_ROWS
        );
        const truncate = (s: string, max: number) =>
          s.length > max ? s.slice(0, max - 1) + "…" : s;

        return (
          <Box flexDirection="column" padding={1}>
            <Box marginBottom={1}>
              <Text bold color="cyan">
                {selectedModel} — Test Results
              </Text>
              <Text dimColor> ({modelResults.length} tests)</Text>
            </Box>

            {modelResults.length === 0 ? (
              <Text color="yellow">No results found for this model.</Text>
            ) : (
              <>
                {/* Results Table */}
                <Table
                  columns={[
                    { key: "id", label: "ID", width: 8 },
                    { key: "input", label: "Input", width: 25 },
                    { key: "expected", label: "Expected", width: 20 },
                    { key: "actual", label: "Actual", width: 20 },
                    { key: "status", label: "Status", width: 8 },
                  ]}
                  data={visibleRows.map((r) => ({
                    id: truncate(String(r.id || ""), 8),
                    input: truncate(String(r.input || ""), 25),
                    expected: truncate(String(r.expected_output || ""), 20),
                    actual: truncate(String(r.actual_output || ""), 20),
                    status: r.passed ? "Pass" : "Fail",
                  }))}
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

                {/* Reasoning for visible rows */}
                <Box marginTop={1} flexDirection="column">
                  <Text bold dimColor>
                    Test Reasoning:
                  </Text>
                  {visibleRows.map((r, idx) => {
                    const reason = String(r.reason || "");
                    if (!reason || reason === "-") return null;
                    return (
                      <Box key={idx} marginTop={1} flexDirection="column">
                        <Box>
                          <Text color={r.passed ? "green" : "red"}>
                            [{String(r.id || idx + 1)}]{" "}
                          </Text>
                          <Text color={r.passed ? "green" : "red"}>
                            {r.passed ? "Pass" : "Fail"}
                          </Text>
                        </Box>
                        <Box marginLeft={2}>
                          <Text wrap="wrap">{reason}</Text>
                        </Box>
                      </Box>
                    );
                  })}
                </Box>
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
            <Text color="red">No evaluation results found.</Text>
            <Box marginTop={1}>
              <Text dimColor>Press q to exit</Text>
            </Box>
          </Box>
        );
      }

      // Sort by pass rate for charts
      const sortedMetrics = [...metrics].sort(
        (a, b) => b.pass_rate - a.pass_rate
      );

      return (
        <Box flexDirection="column" padding={1}>
          <Box marginBottom={1}>
            <Text bold color="cyan">
              LLM Tests Leaderboard
            </Text>
          </Box>

          {/* Comparison Table */}
          <Table
            columns={[
              { key: "model", label: "Model", width: 28 },
              { key: "passed", label: "Passed", width: 8, align: "right" },
              { key: "failed", label: "Failed", width: 8, align: "right" },
              { key: "total", label: "Total", width: 8, align: "right" },
              {
                key: "pass_rate",
                label: "Pass Rate",
                width: 10,
                align: "right",
              },
            ]}
            data={metrics.map((m) => ({
              model: m.model,
              passed: String(m.passed),
              failed: String(m.failed),
              total: String(m.total),
              pass_rate: m.pass_rate.toFixed(1) + "%",
            }))}
          />

          {/* Pass Rate Chart */}
          <Box marginTop={1} flexDirection="column">
            <Text bold>Pass Rate</Text>
            <BarChart
              data={sortedMetrics.map((m) => ({
                label: m.model.length > 25 ? m.model.slice(-25) : m.model,
                value: m.pass_rate,
                color: "green",
              }))}
              maxWidth={40}
            />
          </Box>

          {/* Overall Score Chart if available */}
          {metrics.some((m) => m.overall !== undefined) && (
            <Box marginTop={1} flexDirection="column">
              <Text bold>Overall Score</Text>
              <BarChart
                data={[...metrics]
                  .filter((m) => m.overall !== undefined)
                  .sort((a, b) => (b.overall || 0) - (a.overall || 0))
                  .map((m) => ({
                    label: m.model.length > 25 ? m.model.slice(-25) : m.model,
                    value: m.overall || 0,
                    color: "cyan",
                  }))}
                maxWidth={40}
              />
            </Box>
          )}

          {/* Model selection to view details */}
          <Box marginTop={1} flexDirection="column">
            <Text dimColor>{"\u2500".repeat(50)}</Text>
            <Box marginTop={1}>
              <Text bold>View Model Details</Text>
            </Box>
            <Box marginTop={1}>
              <SelectInput
                items={[
                  ...config.models.map((m) => ({
                    label: `${m} — View test-by-test results`,
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

          {/* Output file paths */}
          <Box marginTop={1} flexDirection="column">
            <Text dimColor>{"\u2500".repeat(50)}</Text>
            <Box marginTop={1} flexDirection="column">
              <Text bold>Output Files</Text>
              <Box>
                <Text>{"  Results:     "}</Text>
                <Text color="cyan">
                  {resolvedOutputDir}/{"<model>"}/results.json
                </Text>
              </Box>
              <Box>
                <Text>{"  Logs:        "}</Text>
                <Text color="cyan">
                  {resolvedOutputDir}/{"<model>"}/results.log
                </Text>
              </Box>
              {metrics.length > 0 && (
                <>
                  <Box>
                    <Text>{"  Leaderboard: "}</Text>
                    <Text color="cyan">
                      {path.resolve(leaderboardDir)}/llm_leaderboard.xlsx
                    </Text>
                  </Box>
                  <Box>
                    <Text>{"  Charts:      "}</Text>
                    <Text color="cyan">{path.resolve(leaderboardDir)}/</Text>
                  </Box>
                </>
              )}
            </Box>
          </Box>
        </Box>
      );
    }

    default:
      return null;
  }
}
