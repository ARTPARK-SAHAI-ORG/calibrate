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
type LlmStep =
  | "init"
  | "config-path"
  | "provider"
  | "enter-model"
  | "model-confirm"
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

interface HistoryMessage {
  role: string;
  content: string;
}

interface ToolCall {
  tool: string;
  arguments: Record<string, unknown>;
}

interface TestResult {
  id: string;
  history: HistoryMessage[];
  evaluationType: string;
  evaluationCriteria: string;
  actualOutput: string;
  passed: boolean;
  reasoning: string;
}

const MAX_PARALLEL_MODELS = 2;

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
  const [modelInput, setModelInput] = useState("");
  const [outputInput, setOutputInput] = useState("./out");

  // ── duplicate model error state ──
  const [duplicateError, setDuplicateError] = useState("");

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
      case "enter-model":
        setStep("provider");
        setDuplicateError("");
        break;
      case "model-confirm":
        setStep("enter-model");
        break;
      case "output-dir":
        setStep("model-confirm");
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
    // Always need OPENAI_API_KEY for LLM judge
    if (!getCredential("OPENAI_API_KEY")) {
      needed.push("OPENAI_API_KEY");
    }
    // Need OPENROUTER_API_KEY if using OpenRouter
    if (provider === "openrouter") {
      if (!getCredential("OPENROUTER_API_KEY")) {
        needed.push("OPENROUTER_API_KEY");
      }
    }
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

  // ── Format tool calls as string ──
  const formatToolCalls = (toolCalls: ToolCall[]): string => {
    if (!toolCalls || toolCalls.length === 0) return "";
    return toolCalls
      .map((tc) => `${tc.tool}(${JSON.stringify(tc.arguments)})`)
      .join(", ");
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
                history?: HistoryMessage[];
                evaluation?: {
                  type?: string;
                  criteria?: string;
                  tool_calls?: ToolCall[];
                };
              };
              output?: {
                response?: string;
                tool_calls?: ToolCall[];
              };
              metrics?: { passed?: boolean; reasoning?: string };
            },
            idx: number
          ) => {
            // Build actual output string from response or tool calls
            let actualOutput = "";
            if (r.output?.response) {
              actualOutput = r.output.response;
            } else if (r.output?.tool_calls && r.output.tool_calls.length > 0) {
              actualOutput = formatToolCalls(r.output.tool_calls);
            }

            // Build evaluation criteria string
            let evaluationCriteria = "";
            const evalType = r.test_case?.evaluation?.type || "";
            if (evalType === "response") {
              evaluationCriteria = r.test_case?.evaluation?.criteria || "";
            } else if (
              evalType === "tool_call" &&
              r.test_case?.evaluation?.tool_calls
            ) {
              evaluationCriteria = formatToolCalls(
                r.test_case.evaluation.tool_calls
              );
            }

            return {
              id: r.test_case?.id || String(idx + 1),
              history: r.test_case?.history || [],
              evaluationType: evalType,
              evaluationCriteria,
              actualOutput,
              passed: r.metrics?.passed || false,
              reasoning: r.metrics?.reasoning || "",
            };
          }
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
      const examples =
        config.provider === "openai"
          ? OPENAI_MODEL_EXAMPLES
          : OPENROUTER_MODEL_EXAMPLES;
      const platformName =
        config.provider === "openai" ? "OpenAI" : "OpenRouter";
      const platformUrl =
        config.provider === "openai"
          ? "platform.openai.com"
          : "openrouter.ai/models";
      const defaultModel =
        config.provider === "openai" ? "gpt-4.1" : "openai/gpt-4.1";

      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Box marginBottom={1}>
            <Text dimColor>Provider: {config.provider}</Text>
          </Box>
          <Text dimColor>
            Enter model name exactly as it appears on {platformName} (
            {platformUrl}).
          </Text>
          {config.models.length > 0 && (
            <Box marginTop={1} flexDirection="column">
              <Text bold>Selected models:</Text>
              {config.models.map((m, i) => (
                <Text key={i} color="green">
                  {"  "}• {m}
                </Text>
              ))}
            </Box>
          )}
          <Box marginTop={1} flexDirection="column">
            <Text dimColor>Examples: {examples.join(", ")}</Text>
          </Box>
          {duplicateError && (
            <Box marginTop={1}>
              <Text color="red">{duplicateError}</Text>
            </Box>
          )}
          <Box marginTop={1}>
            <Text>
              {config.models.length === 0 ? "Model: " : "Add another model: "}
            </Text>
            <TextInput
              value={modelInput}
              onChange={(v) => {
                setModelInput(v);
                setDuplicateError("");
              }}
              onSubmit={(v) => {
                const input = v.trim();
                if (input) {
                  // Check for duplicate
                  if (config.models.includes(input)) {
                    setDuplicateError(`Model "${input}" is already selected.`);
                    return;
                  }
                  // Add the model to the list
                  setConfig((c) => ({
                    ...c,
                    models: [...c.models, input],
                  }));
                  setModelInput("");
                  setDuplicateError("");
                  // Go to confirmation step
                  setStep("model-confirm");
                } else if (config.models.length === 0) {
                  // If no input and no models yet, use default and go to confirmation
                  setConfig((c) => ({
                    ...c,
                    models: [defaultModel],
                  }));
                  setStep("model-confirm");
                } else {
                  // If no input but models already selected, go to confirmation
                  setStep("model-confirm");
                }
              }}
              placeholder={
                config.models.length === 0
                  ? defaultModel
                  : "Enter model name or press Enter to continue"
              }
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>Enter to submit, Esc to go back</Text>
          </Box>
        </Box>
      );
    }

    case "model-confirm": {
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Box flexDirection="column">
            <Text bold>Selected models:</Text>
            {config.models.map((m, i) => (
              <Text key={i} color="green">
                {"  "}• {m}
              </Text>
            ))}
          </Box>
          <Box marginTop={1}>
            <Text>Add another model?</Text>
          </Box>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "Yes, add another model", value: "add" },
                { label: "No, continue with these models", value: "continue" },
              ]}
              onSelect={(v) => {
                if (v === "add") {
                  setStep("enter-model");
                } else {
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

        // Format history messages as "Role: Message"
        const formatHistory = (history: HistoryMessage[]): string => {
          if (!history || history.length === 0) return "-";
          return history.map((h) => `${h.role}: ${h.content}`).join(" | ");
        };

        // Truncate text for display
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
                {/* Per-test results - one block per test */}
                {visibleRows.map((r, idx) => (
                  <Box
                    key={idx}
                    flexDirection="column"
                    marginBottom={1}
                    borderStyle="single"
                    borderColor={r.passed ? "green" : "red"}
                    paddingX={1}
                  >
                    {/* Header with test ID and pass/fail */}
                    <Box marginBottom={1}>
                      <Text bold>Test {r.id}</Text>
                      <Text> </Text>
                      {r.passed ? (
                        <Text color="green" bold>
                          ✓ Pass
                        </Text>
                      ) : (
                        <Text color="red" bold>
                          ✗ Fail
                        </Text>
                      )}
                    </Box>

                    {/* History */}
                    <Box flexDirection="column" marginBottom={1}>
                      <Text bold dimColor>
                        History:
                      </Text>
                      {r.history && r.history.length > 0 ? (
                        r.history.map((h, hIdx) => (
                          <Box key={hIdx} marginLeft={1}>
                            <Text color="cyan">{h.role}: </Text>
                            <Text wrap="wrap">{truncate(h.content, 60)}</Text>
                          </Box>
                        ))
                      ) : (
                        <Box marginLeft={1}>
                          <Text dimColor>-</Text>
                        </Box>
                      )}
                    </Box>

                    {/* Criteria */}
                    <Box flexDirection="column" marginBottom={1}>
                      <Text bold dimColor>
                        Criteria ({r.evaluationType || "unknown"}):
                      </Text>
                      <Box marginLeft={1}>
                        <Text wrap="wrap">
                          {truncate(r.evaluationCriteria || "-", 80)}
                        </Text>
                      </Box>
                    </Box>

                    {/* Actual Output */}
                    <Box flexDirection="column" marginBottom={1}>
                      <Text bold dimColor>
                        Actual Output:
                      </Text>
                      <Box marginLeft={1}>
                        <Text wrap="wrap">
                          {truncate(r.actualOutput || "-", 80)}
                        </Text>
                      </Box>
                    </Box>

                    {/* Reasoning */}
                    <Box flexDirection="column">
                      <Text bold dimColor>
                        Reasoning:
                      </Text>
                      <Box marginLeft={1}>
                        <Text wrap="wrap" color={r.passed ? "green" : "red"}>
                          {r.reasoning || "-"}
                        </Text>
                      </Box>
                    </Box>
                  </Box>
                ))}

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
