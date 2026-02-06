import React, { useState, useEffect, useMemo, useRef } from "react";
import { Box, Text, useApp, useInput } from "ink";
import { spawn, execSync, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import {
  TTS_PROVIDERS,
  LANGUAGES,
  getProviderById,
  getProvidersForLanguage,
} from "./providers.js";
import { getCredential, saveCredential } from "./credentials.js";
import {
  MultiSelect,
  SelectInput,
  TextInput,
  Spinner,
  Table,
  BarChart,
} from "./components.js";

// ─── Resolve calibrate binary ────────────────────────────────
function findCalibrateBin(): { cmd: string; args: string[] } | null {
  // 1. Check if calibrate is in PATH
  try {
    execSync("which calibrate", { stdio: "pipe" });
    return { cmd: "calibrate", args: [] };
  } catch {}

  // 2. Check project .venv (run from ui/ or project root)
  for (const rel of ["../.venv/bin/calibrate", ".venv/bin/calibrate"]) {
    const abs = path.resolve(rel);
    if (fs.existsSync(abs)) {
      return { cmd: abs, args: [] };
    }
  }

  // 3. Check if uv is available and can run calibrate
  try {
    execSync("uv run which calibrate", {
      stdio: "pipe",
      cwd: path.resolve(".."),
    });
    return { cmd: "uv", args: ["run", "calibrate"] };
  } catch {}

  return null;
}

// ─── Types ───────────────────────────────────────────────────
interface CalibrateCmd {
  cmd: string;
  args: string[];
}

interface AppConfig {
  providers: string[];
  inputFile: string;
  language: string;
  outputDir: string;
  envVars: Record<string, string>;
  calibrate: CalibrateCmd;
}

type Step =
  | "config-language"
  | "select-providers"
  | "config-file"
  | "config-output"
  | "setup-keys"
  | "running"
  | "leaderboard";

interface ProviderState {
  status: "waiting" | "running" | "done" | "error";
  logs: string[];
  metrics?: {
    llm_judge_score: number;
    ttfb: { mean: number; std: number; values: number[] };
  };
}

// ─── Helpers ─────────────────────────────────────────────────
function stripAnsi(str: string): string {
  return str.replace(/\x1b\[[0-9;]*m/g, "");
}

// ═════════════════════════════════════════════════════════════
// Step 2: Select TTS Providers (filtered by language)
// ═════════════════════════════════════════════════════════════
function ProviderSelectStep({
  language,
  onComplete,
}: {
  language: string;
  onComplete: (providers: string[]) => void;
}) {
  const availableProviders = useMemo(
    () => getProvidersForLanguage(language),
    [language]
  );

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Calibrate
        </Text>
        <Text bold> — TTS Evaluation</Text>
      </Box>
      <Box marginBottom={1}>
        <Text dimColor>Language: {language}</Text>
      </Box>
      <Text>
        Select providers to evaluate{" "}
        <Text dimColor>
          ({availableProviders.length}/{TTS_PROVIDERS.length} support {language}
          )
        </Text>
      </Text>
      <Box marginTop={1}>
        <MultiSelect
          items={availableProviders.map((p) => ({
            label: p.name,
            value: p.id,
          }))}
          onSubmit={onComplete}
        />
      </Box>
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Step 3: Input CSV file path
// ═════════════════════════════════════════════════════════════
function ConfigFileStep({
  onComplete,
}: {
  onComplete: (file: string) => void;
}) {
  const [value, setValue] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    const candidates = [
      "./examples/tts/sample.csv",
      "../examples/tts/sample.csv",
      "examples/tts/sample.csv",
    ];
    for (const c of candidates) {
      if (fs.existsSync(c)) {
        setValue(c);
        return;
      }
    }
  }, []);

  const handleSubmit = (val: string) => {
    const trimmed = val.trim();
    if (!trimmed) return;
    if (!fs.existsSync(trimmed)) {
      setError(`File not found: ${trimmed}`);
      return;
    }
    if (!trimmed.toLowerCase().endsWith(".csv")) {
      setError("Input must be a CSV file");
      return;
    }
    onComplete(trimmed);
  };

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Configuration
        </Text>
      </Box>
      <Box>
        <Text>Input CSV: </Text>
        <TextInput
          value={value}
          onChange={(v) => {
            setValue(v);
            setError("");
          }}
          onSubmit={handleSubmit}
        />
      </Box>
      {error ? (
        <Box marginTop={1}>
          <Text color="red">{error}</Text>
        </Box>
      ) : (
        <Box marginTop={1}>
          <Text dimColor>
            CSV file with id and text columns. Press enter to confirm.
          </Text>
        </Box>
      )}
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Step 4: Output directory (optional, defaults to ./out)
// ═════════════════════════════════════════════════════════════
function ConfigOutputStep({
  onComplete,
}: {
  onComplete: (dir: string) => void;
}) {
  const [value, setValue] = useState("./out");

  const handleSubmit = (val: string) => {
    const trimmed = val.trim() || "./out";
    onComplete(trimmed);
  };

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Configuration
        </Text>
      </Box>
      <Box>
        <Text>Output directory: </Text>
        <TextInput value={value} onChange={setValue} onSubmit={handleSubmit} />
      </Box>
      <Box marginTop={1}>
        <Text dimColor>Press enter to use default (./out)</Text>
      </Box>
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Step 1: Language selection
// ═════════════════════════════════════════════════════════════
function ConfigLanguageStep({
  onComplete,
}: {
  onComplete: (lang: string) => void;
}) {
  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Calibrate
        </Text>
        <Text bold> — TTS Evaluation</Text>
      </Box>
      <Text>Select language:</Text>
      <Box marginTop={1}>
        <SelectInput
          items={LANGUAGES.map((l) => ({ label: l, value: l }))}
          onSelect={onComplete}
          initialIndex={0}
        />
      </Box>
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Step 3: API Key Setup
// ═════════════════════════════════════════════════════════════
function KeySetupStep({
  selectedProviders,
  onComplete,
}: {
  selectedProviders: string[];
  onComplete: (env: Record<string, string>) => void;
}) {
  // Build list of all needed env vars
  const allKeys = useMemo(() => {
    const result: Array<{
      envVar: string;
      label: string;
      isFilePath?: boolean;
      found: boolean;
    }> = [];
    const seen = new Set<string>();

    // Always need OPENAI_API_KEY for LLM judge
    result.push({
      envVar: "OPENAI_API_KEY",
      label: "OpenAI (LLM Judge)",
      isFilePath: false,
      found: !!getCredential("OPENAI_API_KEY"),
    });
    seen.add("OPENAI_API_KEY");

    for (const id of selectedProviders) {
      const p = getProviderById(id);
      if (p && !seen.has(p.envVar)) {
        result.push({
          envVar: p.envVar,
          label: p.name,
          isFilePath: p.isFilePath,
          found: !!getCredential(p.envVar),
        });
        seen.add(p.envVar);
      }
    }

    return result;
  }, [selectedProviders]);

  const missingKeys = allKeys.filter((k) => !k.found);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [inputValue, setInputValue] = useState("");
  const [enteredKeys, setEnteredKeys] = useState<Set<string>>(new Set());
  const completedRef = useRef(false);

  function buildEnvVars(): Record<string, string> {
    const env: Record<string, string> = {};
    for (const k of allKeys) {
      const val = getCredential(k.envVar);
      if (val) env[k.envVar] = val;
    }
    return env;
  }

  // If no missing keys, auto-complete
  useEffect(() => {
    if (missingKeys.length === 0 && !completedRef.current) {
      completedRef.current = true;
      const timer = setTimeout(() => onComplete(buildEnvVars()), 600);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, []);

  const handleSubmit = (value: string) => {
    if (!value.trim()) return;
    const key = missingKeys[currentIdx]!;
    saveCredential(key.envVar, value.trim());
    setEnteredKeys((prev) => new Set([...prev, key.envVar]));
    setInputValue("");

    if (currentIdx + 1 >= missingKeys.length) {
      setCurrentIdx(currentIdx + 1);
      completedRef.current = true;
      setTimeout(() => onComplete(buildEnvVars()), 400);
    } else {
      setCurrentIdx(currentIdx + 1);
    }
  };

  const allEntered = currentIdx >= missingKeys.length;
  const currentKey = !allEntered ? missingKeys[currentIdx] : null;

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          API Key Setup
        </Text>
      </Box>

      {allKeys.map((key) => {
        const available = key.found || enteredKeys.has(key.envVar);
        const isCurrent = currentKey?.envVar === key.envVar;

        return (
          <Box key={key.envVar}>
            <Text color={available ? "green" : isCurrent ? "cyan" : "gray"}>
              {available ? " + " : isCurrent ? " > " : " - "}
            </Text>
            <Box width={38}>
              <Text color={isCurrent ? "cyan" : undefined} bold={isCurrent}>
                {key.envVar}
              </Text>
            </Box>
            <Text dimColor>
              {key.found
                ? "(stored)"
                : enteredKeys.has(key.envVar)
                ? "(saved)"
                : ""}
            </Text>
          </Box>
        );
      })}

      {currentKey && (
        <Box marginTop={1}>
          <Text>{currentKey.isFilePath ? "Path" : "Key"} for </Text>
          <Text bold color="cyan">
            {currentKey.label}
          </Text>
          <Text>: </Text>
          <TextInput
            value={inputValue}
            onChange={setInputValue}
            onSubmit={handleSubmit}
            mask={currentKey.isFilePath ? undefined : "*"}
            placeholder={
              currentKey.isFilePath
                ? "/path/to/credentials.json"
                : "Enter API key..."
            }
          />
        </Box>
      )}

      {allEntered && missingKeys.length > 0 && (
        <Box marginTop={1}>
          <Text color="green">+ All keys configured!</Text>
        </Box>
      )}

      {missingKeys.length === 0 && (
        <Box marginTop={1}>
          <Text color="green">+ All API keys already configured!</Text>
        </Box>
      )}

      <Box marginTop={1}>
        <Text dimColor>Keys are stored in ~/.calibrate/credentials.json</Text>
      </Box>
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Step 4: Running Evaluations
// ═════════════════════════════════════════════════════════════
function RunStep({
  config,
  onComplete,
}: {
  config: AppConfig;
  onComplete: () => void;
}) {
  const [states, setStates] = useState<Record<string, ProviderState>>(() => {
    const s: Record<string, ProviderState> = {};
    for (const p of config.providers) {
      s[p] = { status: "waiting", logs: [] };
    }
    return s;
  });
  const [currentIdx, setCurrentIdx] = useState(0);
  const [phase, setPhase] = useState<"eval" | "leaderboard" | "done">("eval");
  const processRef = useRef<ChildProcess | null>(null);

  // Run providers sequentially via currentIdx changes
  useEffect(() => {
    if (phase !== "eval") return undefined;
    if (currentIdx >= config.providers.length) {
      setPhase("leaderboard");
      return undefined;
    }

    const provider = config.providers[currentIdx]!;
    setStates((prev) => ({
      ...prev,
      [provider]: { ...prev[provider]!, status: "running" },
    }));

    const proc = spawn(
      config.calibrate.cmd,
      [
        ...config.calibrate.args,
        "tts",
        "eval",
        "-p",
        provider,
        "-l",
        config.language,
        "-i",
        config.inputFile,
        "-o",
        config.outputDir,
      ],
      {
        env: { ...process.env, ...config.envVars },
        stdio: ["pipe", "pipe", "pipe"],
      }
    );

    processRef.current = proc;

    const onData = (data: Buffer) => {
      const lines = data
        .toString()
        .split(/[\r\n]+/)
        .filter((l) => l.trim());
      setStates((prev) => ({
        ...prev,
        [provider]: {
          ...prev[provider]!,
          logs: [...prev[provider]!.logs, ...lines].slice(-12),
        },
      }));
    };

    proc.stdout?.on("data", onData);
    proc.stderr?.on("data", onData);

    proc.on("error", () => {
      setStates((prev) => ({
        ...prev,
        [provider]: { ...prev[provider]!, status: "error" },
      }));
      setCurrentIdx((prev) => prev + 1);
    });

    proc.on("close", (code) => {
      let metrics: ProviderState["metrics"] = undefined;
      if (code === 0) {
        try {
          const metricsPath = path.join(
            config.outputDir,
            provider,
            "metrics.json"
          );
          const raw = JSON.parse(fs.readFileSync(metricsPath, "utf-8"));
          metrics = {
            llm_judge_score: raw.llm_judge_score ?? 0,
            ttfb: raw.ttfb ?? { mean: 0, std: 0, values: [] },
          };
        } catch {
          // metrics might not exist yet
        }
      }

      setStates((prev) => ({
        ...prev,
        [provider]: {
          ...prev[provider]!,
          status: code === 0 ? "done" : "error",
          metrics,
        },
      }));
      setCurrentIdx((prev) => prev + 1);
    });

    return () => {
      proc.kill();
    };
  }, [currentIdx, phase]);

  // Run leaderboard after all evals
  useEffect(() => {
    if (phase !== "leaderboard") return undefined;

    const leaderboardDir = path.join(config.outputDir, "leaderboard");
    const proc = spawn(
      config.calibrate.cmd,
      [
        ...config.calibrate.args,
        "tts",
        "leaderboard",
        "-o",
        config.outputDir,
        "-s",
        leaderboardDir,
      ],
      {
        env: { ...process.env, ...config.envVars },
        stdio: ["pipe", "pipe", "pipe"],
      }
    );

    proc.on("close", () => {
      setPhase("done");
      setTimeout(() => onComplete(), 500);
    });

    proc.on("error", () => {
      setPhase("done");
      setTimeout(() => onComplete(), 500);
    });

    return () => {
      proc.kill();
    };
  }, [phase]);

  const completedCount = Object.values(states).filter(
    (s) => s.status === "done" || s.status === "error"
  ).length;
  const currentProvider =
    currentIdx < config.providers.length ? config.providers[currentIdx] : null;

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          TTS Evaluation
        </Text>
        <Text dimColor>
          {"  "}
          {completedCount}/{config.providers.length} providers
        </Text>
      </Box>

      {/* Provider status list */}
      {config.providers.map((provider) => {
        const state = states[provider]!;
        return (
          <Box key={provider}>
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
            <Box width={15}>
              <Text bold={state.status === "running"}>{provider}</Text>
            </Box>
            {state.status === "done" && state.metrics ? (
              <Text dimColor>
                llm_judge: {state.metrics.llm_judge_score?.toFixed(2)}
                {"  "}ttfb: {state.metrics.ttfb?.mean?.toFixed(2)}s
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

      {/* Log window for current running provider */}
      {phase === "eval" &&
        currentProvider &&
        states[currentProvider]?.status === "running" && (
          <Box flexDirection="column" marginTop={1}>
            <Box>
              <Text dimColor>{"── "}</Text>
              <Text bold>{currentProvider}</Text>
              <Text dimColor>
                {" " +
                  "\u2500".repeat(Math.max(0, 42 - currentProvider.length))}
              </Text>
            </Box>
            <Box flexDirection="column" paddingLeft={1}>
              {(states[currentProvider]?.logs || [])
                .slice(-8)
                .map((line, i) => (
                  <Text key={i} dimColor wrap="truncate">
                    {stripAnsi(line)}
                  </Text>
                ))}
            </Box>
          </Box>
        )}

      {phase === "leaderboard" && (
        <Box marginTop={1}>
          <Spinner label="Generating leaderboard..." />
        </Box>
      )}

      {phase === "done" && (
        <Box marginTop={1}>
          <Text color="green">+ All evaluations complete!</Text>
        </Box>
      )}
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Step 5: Leaderboard Display
// ═════════════════════════════════════════════════════════════
function LeaderboardStep({ config }: { config: AppConfig }) {
  const { exit } = useApp();
  const [metrics, setMetrics] = useState<
    Array<{
      provider: string;
      llm_judge_score: number;
      ttfb: number;
      count: number;
    }>
  >([]);

  useEffect(() => {
    const results: typeof metrics = [];
    for (const provider of config.providers) {
      try {
        const metricsPath = path.join(
          config.outputDir,
          provider,
          "metrics.json"
        );
        const data = JSON.parse(fs.readFileSync(metricsPath, "utf-8"));
        const resultsPath = path.join(
          config.outputDir,
          provider,
          "results.csv"
        );
        let count = 0;
        try {
          const csvContent = fs.readFileSync(resultsPath, "utf-8");
          count = csvContent.trim().split("\n").length - 1;
        } catch {
          // no results.csv
        }

        results.push({
          provider,
          llm_judge_score: data.llm_judge_score ?? 0,
          ttfb: data.ttfb?.mean ?? 0,
          count,
        });
      } catch {
        // skip providers with no metrics
      }
    }
    setMetrics(results);
  }, []);

  useInput((input) => {
    if (input === "q") {
      exit();
    }
  });

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

  const leaderboardDir = path.join(config.outputDir, "leaderboard");
  const resolvedOutputDir = path.resolve(config.outputDir);

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          TTS Leaderboard
        </Text>
      </Box>

      {/* Comparison Table */}
      <Table
        columns={[
          { key: "provider", label: "Provider", width: 14 },
          { key: "llm_judge", label: "LLM Judge", width: 10, align: "right" },
          { key: "ttfb", label: "TTFB (avg)", width: 11, align: "right" },
          { key: "count", label: "Count", width: 6, align: "right" },
        ]}
        data={metrics.map((m) => ({
          provider: m.provider,
          llm_judge: m.llm_judge_score.toFixed(2),
          ttfb: m.ttfb.toFixed(2) + "s",
          count: String(m.count),
        }))}
      />

      {/* LLM Judge Score bar chart */}
      <Box marginTop={1} flexDirection="column">
        <Text bold>LLM Judge Score</Text>
        <BarChart
          data={metrics.map((m) => ({
            label: m.provider,
            value: m.llm_judge_score,
            color: "green",
          }))}
        />
      </Box>

      {/* TTFB bar chart (sorted ascending - lower is better) */}
      <Box marginTop={1} flexDirection="column">
        <Box>
          <Text bold>TTFB </Text>
          <Text dimColor>(lower is better)</Text>
        </Box>
        <BarChart
          data={[...metrics]
            .sort((a, b) => a.ttfb - b.ttfb)
            .map((m) => ({
              label: m.provider,
              value: m.ttfb,
              color: "yellow",
            }))}
        />
      </Box>

      {/* Output file paths */}
      <Box marginTop={1} flexDirection="column">
        <Text dimColor>{"\u2500".repeat(50)}</Text>
        <Box marginTop={1} flexDirection="column">
          <Text bold>Output Files</Text>
          <Box>
            <Text>{"  Audio:       "}</Text>
            <Text color="cyan">
              {resolvedOutputDir}/{"<provider>"}/audios/
            </Text>
          </Box>
          <Box>
            <Text>{"  Results:     "}</Text>
            <Text color="cyan">
              {resolvedOutputDir}/{"<provider>"}/results.csv
            </Text>
          </Box>
          <Box>
            <Text>{"  Leaderboard: "}</Text>
            <Text color="cyan">
              {path.resolve(leaderboardDir)}/tts_leaderboard.xlsx
            </Text>
          </Box>
          <Box>
            <Text>{"  Charts:      "}</Text>
            <Text color="cyan">{path.resolve(leaderboardDir)}/</Text>
          </Box>
        </Box>
      </Box>

      <Box marginTop={1}>
        <Text dimColor>Press q to exit</Text>
      </Box>
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Main App
// ═════════════════════════════════════════════════════════════
export function App() {
  const [step, setStep] = useState<Step | "init">("init");
  const [config, setConfig] = useState<AppConfig>({
    providers: [],
    inputFile: "",
    language: "english",
    outputDir: "./out",
    envVars: {},
    calibrate: { cmd: "calibrate", args: [] },
  });
  const [initError, setInitError] = useState("");

  // Find calibrate binary on mount
  useEffect(() => {
    const result = findCalibrateBin();
    if (result) {
      setConfig((c) => ({ ...c, calibrate: result }));
      setStep("config-language");
    } else {
      setInitError(
        "calibrate CLI not found. Install with: pip install -e . (from project root)"
      );
    }
  }, []);

  if (step === "init" && !initError) {
    return (
      <Box padding={1}>
        <Spinner label="Checking calibrate CLI..." />
      </Box>
    );
  }

  if (initError) {
    return (
      <Box flexDirection="column" padding={1}>
        <Text color="red">x {initError}</Text>
      </Box>
    );
  }

  switch (step) {
    case "config-language":
      return (
        <ConfigLanguageStep
          onComplete={(lang) => {
            setConfig((c) => ({ ...c, language: lang }));
            setStep("select-providers");
          }}
        />
      );

    case "select-providers":
      return (
        <ProviderSelectStep
          language={config.language}
          onComplete={(providers) => {
            setConfig((c) => ({ ...c, providers }));
            setStep("config-file");
          }}
        />
      );

    case "config-file":
      return (
        <ConfigFileStep
          onComplete={(file) => {
            setConfig((c) => ({ ...c, inputFile: file }));
            setStep("config-output");
          }}
        />
      );

    case "config-output":
      return (
        <ConfigOutputStep
          onComplete={(dir) => {
            setConfig((c) => ({ ...c, outputDir: dir }));
            setStep("setup-keys");
          }}
        />
      );

    case "setup-keys":
      return (
        <KeySetupStep
          selectedProviders={config.providers}
          onComplete={(envVars) => {
            setConfig((c) => ({ ...c, envVars }));
            setStep("running");
          }}
        />
      );

    case "running":
      return (
        <RunStep config={config} onComplete={() => setStep("leaderboard")} />
      );

    case "leaderboard":
      return <LeaderboardStep config={config} />;

    default:
      return null;
  }
}
