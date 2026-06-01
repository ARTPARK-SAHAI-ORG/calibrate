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
import {
  type LlmStep,
  type LlmConfig,
  type ModelState,
  type HistoryMessage,
  type ToolCall,
  type TestResult,
  type EvaluatorAggregate,
  MAX_PARALLEL_MODELS,
  OPENAI_MODEL_EXAMPLES,
  OPENROUTER_MODEL_EXAMPLES,
} from "./llm-types.js";

// Parse the per-evaluator aggregate block from a model's metrics.json.
// Returns ``undefined`` when the file is missing/malformed or has no
// ``criteria``/``tool_calls`` blocks. Extracted so both the live progress
// display and the leaderboard view share one parser.
//
// Tool-call aggregates are surfaced alongside response evaluators as columns
// like ``tool:<name>`` (binary, pass-rate %), so the leaderboard table shows
// tool call accuracy next to criteria pass-rates without needing a separate
// section.
function parseEvaluatorsFromMetrics(
  metricsJsonPath: string
): Record<string, EvaluatorAggregate> | undefined {
  try {
    if (!fs.existsSync(metricsJsonPath)) return undefined;
    const mj = JSON.parse(fs.readFileSync(metricsJsonPath, "utf-8"));
    const evaluators: Record<string, EvaluatorAggregate> = {};

    const crit = mj?.criteria;
    if (crit && typeof crit === "object") {
      for (const [name, agg] of Object.entries(
        crit as Record<string, Record<string, unknown>>
      )) {
        if (agg?.["type"] === "binary") {
          const rate =
            typeof agg["pass_rate"] === "number"
              ? (agg["pass_rate"] as number)
              : 0;
          evaluators[name] = {
            type: "binary",
            display: `${rate.toFixed(1)}%`,
            sortValue: rate,
          };
        } else if (agg?.["type"] === "rating") {
          const mean =
            typeof agg["mean"] === "number" ? (agg["mean"] as number) : 0;
          const hi = agg["scale_max"] ?? 5;
          evaluators[name] = {
            type: "rating",
            display: `${mean.toFixed(2)}/${hi}`,
            sortValue: mean,
          };
        }
      }
    }

    const tools = mj?.tool_calls;
    if (tools && typeof tools === "object") {
      for (const [name, agg] of Object.entries(
        tools as Record<string, Record<string, unknown>>
      )) {
        const rate =
          typeof agg?.["pass_rate"] === "number"
            ? (agg["pass_rate"] as number)
            : 0;
        evaluators[`tool:${name}`] = {
          type: "binary",
          display: `${rate.toFixed(1)}%`,
          sortValue: rate,
        };
      }
    }

    return Object.keys(evaluators).length > 0 ? evaluators : undefined;
  } catch {
    return undefined;
  }
}

// Format the evaluators dict as a single inline summary line for a finished
// model row in the live progress display, e.g. ``relevance: 80.0% · tone: 4.20/5 (≥1)``.
// Returns "Done" when no evaluator data is available (e.g. the run had only
// tool_call test cases, which aren't surfaced in the criteria block).
function formatEvaluatorSummary(
  evaluators: Record<string, EvaluatorAggregate> | undefined
): string {
  if (!evaluators || Object.keys(evaluators).length === 0) return "Done";
  return Object.entries(evaluators)
    .map(([name, ev]) => `${name}: ${ev.display}`)
    .join(" · ");
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
    agentUrl: "",
    agentHeaders: {},
    agentBenchmark: false,
    agentModels: [],
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

  // ── agent verify state ──
  const [verifyStatus, setVerifyStatus] = useState<"running" | "success" | "failed">("running");
  const [verifyError, setVerifyError] = useState("");

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
      total?: number;
      passed?: number;
      evaluators?: Record<string, EvaluatorAggregate>;
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
      case "agent-mode":
        setStep("config-path");
        break;
      case "agent-model-entry":
        setStep(config.agentModels.length > 0 ? "agent-model-confirm" : "agent-mode");
        setDuplicateError("");
        break;
      case "agent-model-confirm":
        setStep("agent-verify");
        break;
      case "agent-verify":
        if (config.agentBenchmark) {
          // Remove the last added model and go back to entry
          setConfig((c) => ({ ...c, agentModels: c.agentModels.slice(0, -1) }));
          setStep("agent-model-entry");
        } else {
          setStep("agent-mode");
        }
        break;
      case "output-dir":
        setStep(
          config.agentUrl
            ? config.agentBenchmark ? "agent-model-confirm" : "agent-verify"
            : "model-confirm"
        );
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
    // Always need OPENAI_API_KEY for evaluators
    if (!getCredential("OPENAI_API_KEY")) {
      needed.push("OPENAI_API_KEY");
    }
    // Need OPENROUTER_API_KEY only for internal agent using OpenRouter
    // Agent connection path does not use provider/model, so skip this check
    if (!config.agentUrl && provider === "openrouter") {
      if (!getCredential("OPENROUTER_API_KEY")) {
        needed.push("OPENROUTER_API_KEY");
      }
    }
    return needed;
  }

  // ── Build model directory name (matches Python logic) ──
  function getModelDir(model: string): string {
    const modelDir =
      config.provider === "openai" ? `${config.provider}/${model}` : model;
    return modelDir.replace(/\//g, "__");
  }

  // ── Resolve the directory where results.json lives for a given run key ──
  function getResultsDir(model: string): string {
    // Single agent run: Python saves directly to outputDir (no subfolder)
    if (config.agentUrl && !config.agentBenchmark) {
      return config.outputDir;
    }
    // Benchmark agent run: model name used as subfolder (slashes → __)
    if (config.agentUrl && config.agentBenchmark) {
      return path.join(config.outputDir, model.replace(/\//g, "__"));
    }
    // Internal model path
    return path.join(config.outputDir, getModelDir(model));
  }

  // ── Initialize model states when entering running step ──
  useEffect(() => {
    if (step !== "running") return;

    const initialStates: Record<string, ModelState> = {};
    const keys = config.agentUrl
      ? config.agentBenchmark
        ? config.agentModels
        : ["agent"]
      : config.models;
    for (const key of keys) {
      initialStates[key] = { status: "waiting", logs: [] };
    }
    setModelStates(initialStates);
    setPhase("eval");
    setRunningCount(0);
    setNextModelIdx(0);

    // Clear the shared output-dir `logs` file once at the start of the run.
    // Each per-model subprocess we spawn appends (CALIBRATE_LLM_LOG_APPEND=1)
    // so all model runs end up combined in a single top-level log instead of
    // racing to truncate each other.
    try {
      const sharedLog = path.join(config.outputDir, "logs");
      if (fs.existsSync(sharedLog)) fs.unlinkSync(sharedLog);
    } catch {
      // best-effort: if we cannot clear the file we still proceed; subprocesses
      // will append to whatever's there.
    }
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
    // Tell benchmark.py to append to the shared <output_dir>/logs file instead
    // of truncating it, so concurrent per-model subprocesses don't overwrite
    // each other's output. The UI clears this file once before the run begins.
    env.CALIBRATE_LLM_LOG_APPEND = "1";

    const cmdArgs = config.agentUrl
      ? [
          ...bin.args,
          "llm",
          "-c",
          config.configPath,
          "-o",
          config.outputDir,
          "--skip-verify",
          ...(config.agentBenchmark ? ["-m", model] : []),
        ]
      : [
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
          const resultsPath = path.join(getResultsDir(model), "results.json");
          if (fs.existsSync(resultsPath)) {
            const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
            const passed = results.filter(
              (r: { metrics?: { passed?: boolean } }) => r.metrics?.passed
            ).length;
            const total = results.length;
            metricsData = { passed, failed: total - passed, total };
          }

          // Also load per-evaluator aggregate so the live progress row can
          // surface per-evaluator stats instead of a generic pass count.
          const evaluators = parseEvaluatorsFromMetrics(
            path.join(getResultsDir(model), "metrics.json")
          );
          if (evaluators) {
            metricsData = { ...(metricsData ?? {}), evaluators };
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

    const runKeys = config.agentUrl
      ? config.agentBenchmark
        ? config.agentModels
        : ["agent"]
      : config.models;

    if (completedCount >= runKeys.length) {
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
      nextModelIdx < runKeys.length
    ) {
      const model = runKeys[nextModelIdx]!;
      setNextModelIdx((idx) => idx + 1);
      startModel(model);
    }
  }, [step, phase, runningCount, nextModelIdx, modelStates]);

  // ── Agent verify effect ──
  useEffect(() => {
    if (step !== "agent-verify") return;
    setVerifyStatus("running");
    setVerifyError("");

    const bin = config.calibrate;
    const verifyArgs = [
      ...bin.args,
      "llm",
      "--verify",
      "--agent-url",
      config.agentUrl,
      ...(Object.keys(config.agentHeaders).length > 0
        ? ["--agent-headers", JSON.stringify(config.agentHeaders)]
        : []),
      ...(config.agentBenchmark && config.agentModels.length > 0
        ? ["-m", config.agentModels[config.agentModels.length - 1]!]
        : []),
    ];

    const env: Record<string, string> = { ...process.env } as Record<string, string>;
    env.PYTHONUNBUFFERED = "1";

    const proc = spawn(bin.cmd, verifyArgs, { env, stdio: ["pipe", "pipe", "pipe"] });

    let output = "";
    proc.stdout?.on("data", (d: Buffer) => { output += d.toString(); });
    proc.stderr?.on("data", (d: Buffer) => { output += d.toString(); });

    proc.on("close", (code: number | null) => {
      if (code === 0) {
        setVerifyStatus("success");
        setTimeout(() => {
          setStep(config.agentBenchmark ? "agent-model-confirm" : "output-dir");
        }, 1000);
      } else {
        // Extract error message from output
        const lines = output.split("\n").filter((l) => l.trim());
        const errorLine =
          lines.find((l) => l.includes("✗") || l.includes("Verification failed") || l.includes("error")) ||
          lines[lines.length - 1] ||
          "Connection failed";
        setVerifyStatus("failed");
        setVerifyError(errorLine);
      }
    });

    proc.on("error", () => {
      setVerifyStatus("failed");
      setVerifyError("Failed to run verification");
    });
  }, [step]);

  // ── Load metrics for leaderboard ──
  const loadMetrics = () => {
    const results: typeof metrics = [];
    const metricKeys = config.agentUrl
      ? config.agentBenchmark
        ? config.agentModels
        : ["agent"]
      : config.models;

    for (const model of metricKeys) {
      const resultsPath = path.join(getResultsDir(model), "results.json");
      if (!fs.existsSync(resultsPath)) continue;
      const evaluators = parseEvaluatorsFromMetrics(
        path.join(getResultsDir(model), "metrics.json")
      );

      // Read total/passed counts so the leaderboard can show overall test
      // pass rate alongside the per-evaluator columns. Prefer metrics.json
      // (canonical), fall back to counting results.json entries.
      let total: number | undefined;
      let passed: number | undefined;
      try {
        const metricsPath = path.join(getResultsDir(model), "metrics.json");
        if (fs.existsSync(metricsPath)) {
          const mj = JSON.parse(fs.readFileSync(metricsPath, "utf-8"));
          if (typeof mj?.total === "number") total = mj.total;
          if (typeof mj?.passed === "number") passed = mj.passed;
        }
        if (total === undefined || passed === undefined) {
          const data = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
          if (Array.isArray(data)) {
            total = total ?? data.length;
            passed =
              passed ??
              data.filter(
                (r: { metrics?: { passed?: boolean } }) => r.metrics?.passed
              ).length;
          }
        }
      } catch {
        // Best-effort; missing counts just render as "-".
      }

      results.push({ model, total, passed, evaluators });
    }

    setMetrics(results);
  };

  // ── Format tool calls as string ──
  // ``includeOutput`` appends the tool's own result (``=> <output>``) when the
  // agent supplied one. Only used for actual output, not the expected criteria.
  const formatToolCalls = (
    toolCalls: ToolCall[],
    includeOutput = false
  ): string => {
    if (!toolCalls || toolCalls.length === 0) return "";
    return toolCalls
      .map((tc) => {
        const call = `${tc.tool}(${JSON.stringify(tc.arguments)})`;
        if (includeOutput && tc.output !== undefined) {
          return `${call} => ${JSON.stringify(tc.output)}`;
        }
        return call;
      })
      .join(", ");
  };

  // ── Load model results when selected ──
  useEffect(() => {
    // Always clear previously-loaded results before doing anything else so
    // navigating from a successful model into a failed one never flashes
    // (or persists) the previous model's test data.
    setModelResults([]);
    setScrollOffset(0);
    if (!selectedModel) return;
    try {
      const resultsPath = path.join(getResultsDir(selectedModel), "results.json");
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
                  criteria?: unknown;
                  tool_calls?: ToolCall[];
                };
              };
              output?: {
                response?: string;
                tool_calls?: ToolCall[];
              };
              metrics?: {
                passed?: boolean;
                reasoning?: string;
                judge_results?: Record<
                  string,
                  { match?: boolean; score?: number; reasoning?: string }
                >;
              };
            },
            idx: number
          ) => {
            // Build actual output string from response or tool calls
            let actualOutput = "";
            if (r.output?.response) {
              actualOutput = r.output.response;
            } else if (r.output?.tool_calls && r.output.tool_calls.length > 0) {
              actualOutput = formatToolCalls(r.output.tool_calls, true);
            }

            // Build evaluation criteria string. Under the new evaluators
            // model, ``evaluation.criteria`` is a list of names (strings or
            // ``{name}`` dicts). Old configs used a single string; preserve
            // that as a fallback.
            let evaluationCriteria = "";
            const evalType = r.test_case?.evaluation?.type || "";
            if (evalType === "response") {
              const raw = r.test_case?.evaluation?.criteria;
              if (Array.isArray(raw)) {
                evaluationCriteria = raw
                  .map((c) => {
                    if (typeof c === "string") return c;
                    if (c && typeof c === "object" && "name" in c) {
                      return String((c as { name: unknown }).name);
                    }
                    return "";
                  })
                  .filter(Boolean)
                  .join(", ");
              } else if (typeof raw === "string") {
                evaluationCriteria = raw;
              }
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
              judgeResults: r.metrics?.judge_results,
            };
          }
        );
        setModelResults(results);
      }
    } catch {
      // Leave modelResults empty so the model-detail view falls into the
      // "no results" / failure path rather than displaying stale data.
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
                  let agentUrl = "";
                  let agentHeaders: Record<string, string> = {};
                  try {
                    const parsed = JSON.parse(fs.readFileSync(resolved, "utf-8"));
                    agentUrl = parsed.agent_url || "";
                    agentHeaders = parsed.agent_headers || {};
                  } catch {
                    // ignore parse errors, treat as internal agent
                  }
                  setConfig((c) => ({ ...c, configPath: resolved, agentUrl, agentHeaders }));
                  setStep(agentUrl ? "agent-mode" : "provider");
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

    case "agent-mode":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>How do you want to run tests against your agent?</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "Single test — run all test cases once", value: "single" },
                { label: "Benchmark — run across multiple models", value: "benchmark" },
              ]}
              onSelect={(v) => {
                const isBenchmark = v === "benchmark";
                setConfig((c) => ({ ...c, agentBenchmark: isBenchmark, agentModels: [] }));
                setStep(isBenchmark ? "agent-model-entry" : "agent-verify");
              }}
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>Press Esc to go back</Text>
          </Box>
        </Box>
      );

    case "agent-model-entry": {
      const agentModelInput = modelInput;
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>Enter the model name to benchmark. Your agent will receive the model name in each request.</Text>
          {config.agentModels.length > 0 && (
            <Box marginTop={1} flexDirection="column">
              <Text bold>Selected models:</Text>
              {config.agentModels.map((m, i) => (
                <Text key={i} color="green">{"  "}• {m}</Text>
              ))}
            </Box>
          )}
          {duplicateError && (
            <Box marginTop={1}>
              <Text color="red">{duplicateError}</Text>
            </Box>
          )}
          <Box marginTop={1}>
            <Text>{config.agentModels.length === 0 ? "Model: " : "Add another model: "}</Text>
            <TextInput
              value={agentModelInput}
              onChange={(v) => { setModelInput(v); setDuplicateError(""); }}
              onSubmit={(v) => {
                const input = v.trim();
                if (input) {
                  if (config.agentModels.includes(input)) {
                    setDuplicateError(`Model "${input}" is already selected.`);
                    return;
                  }
                  setConfig((c) => ({ ...c, agentModels: [...c.agentModels, input] }));
                  setModelInput("");
                  setDuplicateError("");
                  setStep("agent-verify");
                } else if (config.agentModels.length > 0) {
                  setStep("agent-model-confirm");
                }
              }}
              placeholder="gemma-4-26b-a4b-it"
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>Enter to submit, Esc to go back</Text>
          </Box>
        </Box>
      );
    }

    case "agent-model-confirm": {
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Box flexDirection="column">
            <Text bold>Selected models:</Text>
            {config.agentModels.map((m, i) => (
              <Text key={i} color="green">{"  "}• {m}</Text>
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
                  setStep("agent-model-entry");
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

    case "agent-verify": {
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>Verifying agent connection: {config.agentUrl}</Text>
          <Box marginTop={1}>
            {verifyStatus === "running" && (
              <Spinner label="Verifying connection..." />
            )}
            {verifyStatus === "success" && (
              <Text color="green">✓ Connection verified</Text>
            )}
            {verifyStatus === "failed" && (
              <Box flexDirection="column">
                <Text color="red">✗ Verification failed</Text>
                {verifyError && (
                  <Box marginTop={1}>
                    <Text dimColor>{stripAnsi(verifyError)}</Text>
                  </Box>
                )}
                <Box marginTop={1}>
                  <SelectInput
                    items={[{ label: "Go back", value: "back" }]}
                    onSelect={() => {
                      if (config.agentBenchmark) {
                        setConfig((c) => ({ ...c, agentModels: c.agentModels.slice(0, -1) }));
                        setStep("agent-model-entry");
                      } else {
                        setStep("agent-mode");
                      }
                    }}
                  />
                </Box>
              </Box>
            )}
          </Box>
        </Box>
      );
    }

    case "running": {
      const completedCount = Object.values(modelStates).filter(
        (s) => s.status === "done" || s.status === "error"
      ).length;

      const runKeys = config.agentUrl
        ? config.agentBenchmark
          ? config.agentModels
          : ["agent"]
        : config.models;

      // Get currently running models for log display
      const runningModels = runKeys.filter(
        (m) => modelStates[m]?.status === "running"
      );

      // Single agent run: simpler status display, no model label
      const isSingleAgentRun = config.agentUrl && !config.agentBenchmark;
      const singleState = isSingleAgentRun ? modelStates["agent"] : undefined;

      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Box marginBottom={1}>
            <Text dimColor>Config: {config.configPath}</Text>
          </Box>
          <Box marginBottom={1}>
            <Text dimColor>
              {config.agentUrl
                ? `Agent: ${config.agentUrl}`
                : `${completedCount}/${runKeys.length} models${runningCount > 1 ? ` (${runningCount} running in parallel)` : ""} | Provider: ${config.provider}`}
            </Text>
          </Box>

          {/* Single agent run: just a status line, no model label */}
          {isSingleAgentRun && singleState && (
            <Box>
              <Box width={4}>
                {singleState.status === "done" ? (
                  <Text color="green"> + </Text>
                ) : singleState.status === "error" ? (
                  <Text color="red"> x </Text>
                ) : (
                  <Box><Text> </Text><Spinner /><Text> </Text></Box>
                )}
              </Box>
              {singleState.status === "done" ? (
                <Text dimColor>{formatEvaluatorSummary(singleState.metrics?.evaluators)}</Text>
              ) : singleState.status === "running" ? (
                <Text color="cyan">Running tests...</Text>
              ) : singleState.status === "error" ? (
                <Text color="red">Failed</Text>
              ) : (
                <Text dimColor>Waiting</Text>
              )}
            </Box>
          )}

          {/* Multi-model status list (benchmark or internal models) */}
          {!isSingleAgentRun && runKeys.map((model) => {
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
                {state.status === "done" ? (
                  <Text dimColor>
                    {formatEvaluatorSummary(state.metrics?.evaluators)}
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

      // Determine the model keys for this run
      const leaderboardKeys = config.agentUrl
        ? config.agentBenchmark
          ? config.agentModels
          : ["agent"]
        : config.models;

      // Classify each leaderboard key as succeeded vs failed using the
      // ground truth on disk: a successful run always writes results.json.
      // We also fall back to modelStates for the in-session error string.
      const isModelFailed = (m: string): boolean => {
        const resultsPath = path.join(getResultsDir(m), "results.json");
        if (fs.existsSync(resultsPath)) return false;
        // No results.json — either still running (handled elsewhere) or
        // the subprocess errored out before writing any.
        return modelStates[m]?.status !== "running";
      };

      // Model Detail View
      if (view === "model-detail" && selectedModel) {
        const failed = isModelFailed(selectedModel);

        // Failed model: show captured error output instead of empty test
        // results. We never want to fall through into the test-rendering
        // branch (which would either say "No results found" or — worse —
        // display whichever model the user previously inspected).
        if (failed) {
          const errorLogs = (modelStates[selectedModel]?.logs ?? []).map(
            (l) => stripAnsi(l)
          );
          return (
            <Box flexDirection="column" padding={1}>
              <Box marginBottom={1}>
                <Text bold color="cyan">
                  {selectedModel}
                </Text>
                <Text dimColor> — </Text>
                <Text bold color="red">
                  Run Failed
                </Text>
              </Box>
              <Box marginBottom={1}>
                <Text>
                  No results were produced for this model. The
                  {" "}<Text color="cyan">calibrate llm</Text>{" "}
                  subprocess exited with an error.
                </Text>
              </Box>
              <Box flexDirection="column" marginBottom={1}>
                <Text bold dimColor>
                  Last subprocess output:
                </Text>
                <Box
                  flexDirection="column"
                  marginTop={1}
                  borderStyle="single"
                  borderColor="red"
                  paddingX={1}
                >
                  {errorLogs.length > 0 ? (
                    errorLogs.map((line, i) => (
                      <Text key={i} wrap="wrap">
                        {line}
                      </Text>
                    ))
                  ) : (
                    <Text dimColor>
                      (no output captured — try re-running this model)
                    </Text>
                  )}
                </Box>
              </Box>
              <Box marginTop={1}>
                <Text dimColor>Press q or Esc to go back to leaderboard</Text>
              </Box>
            </Box>
          );
        }

        const visibleRows = modelResults.slice(
          scrollOffset,
          scrollOffset + MAX_VISIBLE_ROWS
        );

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
                    {/* Test ID header with overall pass/fail indicator. */}
                    <Box marginBottom={1}>
                      <Text bold>Test {r.id} </Text>
                      <Text bold color={r.passed ? "green" : "red"}>
                        {r.passed ? "✓ Pass" : "✗ Fail"}
                      </Text>
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

                    {/* Evaluators */}
                    <Box flexDirection="column" marginBottom={1}>
                      <Text bold dimColor>
                        Evaluators ({r.evaluationType || "unknown"}):
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

                    {/* Per-evaluator judge results */}
                    {r.judgeResults &&
                      Object.keys(r.judgeResults).length > 0 && (
                        <Box flexDirection="column" marginBottom={1}>
                          <Text bold dimColor>
                            Evaluator Results:
                          </Text>
                          {Object.entries(r.judgeResults).map(([name, ev]) => {
                            const isBinary = typeof ev?.match === "boolean";
                            const isRating = typeof ev?.score === "number";
                            const color = isBinary
                              ? ev?.match
                                ? "green"
                                : "red"
                              : undefined;
                            const label = isBinary
                              ? ev?.match
                                ? "Pass"
                                : "Fail"
                              : isRating
                              ? String(ev?.score)
                              : "-";
                            return (
                              <Box
                                key={name}
                                flexDirection="column"
                                marginLeft={1}
                                marginTop={1}
                              >
                                <Box>
                                  <Text bold>{name}: </Text>
                                  <Text color={color}>{label}</Text>
                                </Box>
                                {ev?.reasoning ? (
                                  <Box marginLeft={2}>
                                    <Text wrap="wrap">{ev.reasoning}</Text>
                                  </Box>
                                ) : null}
                              </Box>
                            );
                          })}
                        </Box>
                      )}

                    {/* Reasoning */}
                    <Box flexDirection="column">
                      <Text bold dimColor>
                        Summary:
                      </Text>
                      <Box marginLeft={1}>
                        <Text wrap="wrap">{r.reasoning || "-"}</Text>
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

      // Union of evaluator names across models, preserving first-seen order
      // so the table columns and chart sections line up regardless of which
      // model loaded first.
      const seenEval = new Set<string>();
      const evaluatorNames: string[] = [];
      for (const m of metrics) {
        for (const name of Object.keys(m.evaluators ?? {})) {
          if (!seenEval.has(name)) {
            seenEval.add(name);
            evaluatorNames.push(name);
          }
        }
      }

      // Resolve a single evaluator's type/scale info from the first model
      // that reports it. Used to label per-evaluator charts (e.g. add
      // "rating 1-5" to a rating evaluator's title).
      const evaluatorMeta: Record<
        string,
        { type: "binary" | "rating" }
      > = {};
      for (const name of evaluatorNames) {
        const found = metrics.find((m) => m.evaluators?.[name])?.evaluators?.[
          name
        ];
        if (found) evaluatorMeta[name] = { type: found.type };
      }

      return (
        <Box flexDirection="column" padding={1}>
          <Box marginBottom={1}>
            <Text bold color="cyan">
              LLM Tests Leaderboard
            </Text>
          </Box>

          {/* Leaderboard table: overall test pass counts + per-evaluator
              comparison columns. */}
          {(() => {
            // Tool-call aggregates are namespaced as ``tool:<name>``; widen
            // the cap a little so the prefix doesn't eat the visible space.
            const labelCap = 18;
            const evaluatorColumns = evaluatorNames.map((name) => ({
              key: name,
              label: name.length > labelCap ? name.slice(0, labelCap) : name,
              width: Math.max(10, Math.min(name.length, labelCap)),
              align: "right" as const,
            }));
            return (
              <Table
                columns={[
                  { key: "model", label: "Model", width: 28 },
                  { key: "total", label: "Total", width: 7, align: "right" as const },
                  { key: "passed", label: "Passed", width: 8, align: "right" as const },
                  { key: "pass_rate", label: "Pass %", width: 9, align: "right" as const },
                  ...evaluatorColumns,
                ]}
                data={metrics.map((m) => {
                  const row: Record<string, string> = { model: m.model };
                  row.total = m.total != null ? String(m.total) : "-";
                  row.passed = m.passed != null ? String(m.passed) : "-";
                  row.pass_rate =
                    m.total != null && m.passed != null && m.total > 0
                      ? `${((m.passed / m.total) * 100).toFixed(1)}%`
                      : "-";
                  for (const name of evaluatorNames) {
                    row[name] = m.evaluators?.[name]?.display ?? "-";
                  }
                  return row;
                })}
              />
            );
          })()}

          {/* One chart per evaluator: bars compare every model on that
              evaluator alone (binary → pass-rate %, rating → mean score).
              No overall/generic pass-rate chart. */}
          {evaluatorNames.map((name) => {
            const meta = evaluatorMeta[name];
            const data = [...metrics]
              .filter((m) => m.evaluators?.[name])
              .sort(
                (a, b) =>
                  (b.evaluators![name]!.sortValue) -
                  (a.evaluators![name]!.sortValue)
              )
              .map((m) => ({
                label: m.model.length > 25 ? m.model.slice(-25) : m.model,
                value: m.evaluators![name]!.sortValue,
                color: meta?.type === "rating" ? "cyan" : "green",
              }));
            if (data.length === 0) return null;
            const subtitle =
              meta?.type === "rating" ? "(mean rating)" : "(% passed)";
            return (
              <Box marginTop={1} flexDirection="column" key={name}>
                <Box>
                  <Text bold>{name} </Text>
                  <Text dimColor>{subtitle}</Text>
                </Box>
                <BarChart data={data} maxWidth={40} />
              </Box>
            );
          })}

          {/* Model selection to view details */}
          <Box marginTop={1} flexDirection="column">
            <Text dimColor>{"\u2500".repeat(50)}</Text>
            <Box marginTop={1}>
              <Text bold>View Model Details</Text>
            </Box>
            <Box marginTop={1}>
              <SelectInput
                items={[
                  ...leaderboardKeys.map((m) => {
                    const isAgent =
                      config.agentUrl && !config.agentBenchmark;
                    const failed = isModelFailed(m);
                    if (failed) {
                      return {
                        label: isAgent
                          ? "Failed (view error)"
                          : `${m} — Failed (view error)`,
                        value: m,
                      };
                    }
                    return {
                      label: isAgent
                        ? "View test-by-test results"
                        : `${m} — View test-by-test results`,
                      value: m,
                    };
                  }),
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
