import React, { useState, useEffect, useMemo, useRef } from "react";
import { Box, Text, useApp, useInput } from "ink";
import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import {
  TTS_PROVIDERS,
  STT_PROVIDERS,
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
import {
  type AppMode,
  type CalibrateCmd,
  findCalibrateBin,
  stripAnsi,
  findAvailablePort,
} from "./shared.js";
import { LlmTestsApp } from "./llm-app.js";
import { SimulationsApp } from "./sim-app.js";

// ─── Types ───────────────────────────────────────────────────
export type Mode = AppMode;

type EvalMode = "tts" | "stt";

interface EvalConfig {
  mode: EvalMode;
  providers: string[];
  inputPath: string;
  language: string;
  outputDir: string;
  overwrite: boolean;
  envVars: Record<string, string>;
  calibrate: CalibrateCmd;
}

type Step =
  | "config-language"
  | "select-providers"
  | "config-input"
  | "config-output"
  | "setup-keys"
  | "running";

interface ProviderState {
  status: "waiting" | "running" | "done" | "error";
  logs: string[];
  metrics?: Record<string, number>;
}

// ─── Helpers ─────────────────────────────────────────────────

function getModeLabel(mode: EvalMode): string {
  return mode === "tts" ? "TTS" : "STT";
}

function getAllProviders(mode: EvalMode) {
  return mode === "tts" ? TTS_PROVIDERS : STT_PROVIDERS;
}

// ═════════════════════════════════════════════════════════════
// Step 1: Language selection
// ═════════════════════════════════════════════════════════════
function ConfigLanguageStep({
  mode,
  onComplete,
  onBack,
}: {
  mode: EvalMode;
  onComplete: (lang: string) => void;
  onBack?: () => void;
}) {
  useInput((_input, key) => {
    if (key.escape && onBack) {
      onBack();
    }
  });

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Calibrate
        </Text>
        <Text bold> — {getModeLabel(mode)} Evaluation</Text>
      </Box>
      <Text>Select language:</Text>
      <Box marginTop={1}>
        <SelectInput
          items={LANGUAGES.map((l) => ({ label: l, value: l }))}
          onSelect={onComplete}
          initialIndex={0}
        />
      </Box>
      {onBack && (
        <Box marginTop={1}>
          <Text dimColor>Press Esc to go back</Text>
        </Box>
      )}
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Step 2: Select Providers (filtered by language)
// ═════════════════════════════════════════════════════════════
function ProviderSelectStep({
  mode,
  language,
  onComplete,
  onBack,
}: {
  mode: EvalMode;
  language: string;
  onComplete: (providers: string[]) => void;
  onBack: () => void;
}) {
  const allProviders = getAllProviders(mode);
  const availableProviders = useMemo(
    () => getProvidersForLanguage(language, mode),
    [language, mode]
  );

  useInput((_input, key) => {
    if (key.escape) {
      onBack();
    }
  });

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Calibrate
        </Text>
        <Text bold> — {getModeLabel(mode)} Evaluation</Text>
      </Box>
      <Box marginBottom={1}>
        <Text dimColor>Language: {language}</Text>
      </Box>
      <Text>
        Select providers to evaluate{" "}
        <Text dimColor>
          ({availableProviders.length}/{allProviders.length} support {language})
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
      <Box marginTop={1}>
        <Text dimColor>Press Esc to go back</Text>
      </Box>
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Input validation helpers
// ═════════════════════════════════════════════════════════════

function validateTtsInput(inputPath: string): {
  valid: boolean;
  error: string;
} {
  // Check file exists
  if (!fs.existsSync(inputPath)) {
    return { valid: false, error: `File not found: ${inputPath}` };
  }

  // Check it's a CSV file
  if (!inputPath.toLowerCase().endsWith(".csv")) {
    return { valid: false, error: "Input must be a CSV file" };
  }

  // Read and validate CSV structure
  try {
    const content = fs.readFileSync(inputPath, "utf-8");
    const lines = content.trim().split("\n");
    if (lines.length < 2) {
      return { valid: false, error: "CSV file is empty (no data rows)" };
    }

    const header = lines[0]!.toLowerCase();
    if (!header.includes("id")) {
      return { valid: false, error: "CSV missing required column 'id'" };
    }
    if (!header.includes("text")) {
      return { valid: false, error: "CSV missing required column 'text'" };
    }
  } catch (e) {
    return { valid: false, error: `Failed to read CSV: ${e}` };
  }

  return { valid: true, error: "" };
}

function validateSttInput(
  inputDir: string,
  csvFileName: string = "stt.csv"
): { valid: boolean; error: string } {
  // Check directory exists
  if (!fs.existsSync(inputDir)) {
    return { valid: false, error: `Directory not found: ${inputDir}` };
  }

  const stat = fs.statSync(inputDir);
  if (!stat.isDirectory()) {
    return { valid: false, error: "Input must be a directory" };
  }

  // Check CSV file exists
  const csvPath = path.join(inputDir, csvFileName);
  if (!fs.existsSync(csvPath)) {
    return { valid: false, error: `CSV file not found: ${csvPath}` };
  }

  // Check audios directory exists
  const audiosDir = path.join(inputDir, "audios");
  if (!fs.existsSync(audiosDir)) {
    return { valid: false, error: `Audios directory not found: ${audiosDir}` };
  }

  // Read and validate CSV structure
  let ids: string[] = [];
  try {
    const content = fs.readFileSync(csvPath, "utf-8");
    const lines = content.trim().split("\n");
    if (lines.length < 2) {
      return { valid: false, error: "CSV file is empty (no data rows)" };
    }

    const header = lines[0]!.toLowerCase();
    if (!header.includes("id")) {
      return { valid: false, error: "CSV missing required column 'id'" };
    }
    if (!header.includes("text")) {
      return { valid: false, error: "CSV missing required column 'text'" };
    }

    // Parse IDs from CSV (assuming 'id' is first column)
    const headerParts = lines[0]!.split(",").map((h) => h.trim().toLowerCase());
    const idIndex = headerParts.indexOf("id");
    if (idIndex >= 0) {
      ids = lines.slice(1).map((line) => line.split(",")[idIndex]!.trim());
    }
  } catch (e) {
    return { valid: false, error: `Failed to read CSV: ${e}` };
  }

  // Check if all audio files exist
  const missingFiles: string[] = [];
  for (const id of ids) {
    const audioPath = path.join(audiosDir, `${id}.wav`);
    if (!fs.existsSync(audioPath)) {
      missingFiles.push(`${id}.wav`);
    }
  }

  if (missingFiles.length > 0) {
    const shown = missingFiles.slice(0, 3).join(", ");
    const more =
      missingFiles.length > 3 ? ` and ${missingFiles.length - 3} more` : "";
    return { valid: false, error: `Missing audio files: ${shown}${more}` };
  }

  return { valid: true, error: "" };
}

// ═════════════════════════════════════════════════════════════
// Step 3: Input path (CSV file for TTS, directory for STT)
// ═════════════════════════════════════════════════════════════
function ConfigInputStep({
  mode,
  onComplete,
  onBack,
}: {
  mode: EvalMode;
  onComplete: (inputPath: string) => void;
  onBack: () => void;
}) {
  const [value, setValue] = useState("");
  const [error, setError] = useState("");

  useInput((_input, key) => {
    if (key.escape) {
      onBack();
    }
  });

  const handleSubmit = (val: string) => {
    const trimmed = val.trim();
    if (!trimmed) return;

    // Full validation based on mode
    const result =
      mode === "tts" ? validateTtsInput(trimmed) : validateSttInput(trimmed);

    if (!result.valid) {
      setError(result.error);
      return;
    }

    onComplete(trimmed);
  };

  const label = mode === "tts" ? "Input CSV" : "Input directory";
  const hint =
    mode === "tts"
      ? "CSV file with id and text columns. Press enter to confirm."
      : "Directory containing audio files and stt.csv. Press enter to confirm.";
  const docsUrl =
    mode === "tts"
      ? "https://calibrate.artpark.ai/docs/cli/text-to-speech"
      : "https://calibrate.artpark.ai/docs/cli/speech-to-text";

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Configuration
        </Text>
      </Box>
      <Box>
        <Text>{label}: </Text>
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
        <Box marginTop={1} flexDirection="column">
          <Text dimColor>{hint}</Text>
          <Text dimColor>
            See input format: <Text color="blue">{docsUrl}</Text>
          </Text>
        </Box>
      )}
      <Box marginTop={1}>
        <Text dimColor>Press Esc to go back</Text>
      </Box>
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Step 4: Output directory
// ═════════════════════════════════════════════════════════════
function ConfigOutputStep({
  providers,
  onComplete,
  onBack,
}: {
  providers: string[];
  onComplete: (dir: string, overwrite: boolean) => void;
  onBack: () => void;
}) {
  const [value, setValue] = useState("./out");
  const [confirmOverwrite, setConfirmOverwrite] = useState<{
    dir: string;
    existingDirs: string[];
  } | null>(null);

  useInput((_input, key) => {
    if (key.escape && !confirmOverwrite) {
      onBack();
    }
  });

  const checkExistingOutput = (outputDir: string): string[] => {
    const existing: string[] = [];
    for (const provider of providers) {
      const providerDir = path.join(outputDir, provider);
      if (fs.existsSync(providerDir)) {
        try {
          const contents = fs.readdirSync(providerDir);
          if (contents.length > 0) {
            existing.push(provider);
          }
        } catch {
          // Ignore read errors
        }
      }
    }
    return existing;
  };

  const handleSubmit = (val: string) => {
    const trimmed = val.trim() || "./out";

    // Check if any provider output directories already exist
    const existing = checkExistingOutput(trimmed);
    if (existing.length > 0) {
      setConfirmOverwrite({ dir: trimmed, existingDirs: existing });
      return;
    }

    onComplete(trimmed, false);
  };

  const handleOverwriteConfirm = (overwrite: boolean) => {
    if (overwrite && confirmOverwrite) {
      // User confirmed overwrite - pass flag to CLI (don't wipe here)
      onComplete(confirmOverwrite.dir, true);
    } else {
      // User declined, let them enter new path
      setConfirmOverwrite(null);
      setValue("");
    }
  };

  // Confirmation prompt
  if (confirmOverwrite) {
    return (
      <Box flexDirection="column" padding={1}>
        <Box marginBottom={1}>
          <Text bold color="yellow">
            Warning: Existing Output Found
          </Text>
        </Box>
        <Text>The following provider directories already contain data:</Text>
        <Box flexDirection="column" marginLeft={2} marginY={1}>
          {confirmOverwrite.existingDirs.map((dir) => (
            <Text key={dir} color="yellow">
              • {path.join(confirmOverwrite.dir, dir)}
            </Text>
          ))}
        </Box>
        <Text>Do you want to overwrite existing results?</Text>
        <Box marginTop={1}>
          <SelectInput
            items={[
              { label: "Yes, overwrite and continue", value: "yes" },
              { label: "No, enter a different path", value: "no" },
            ]}
            onSelect={(v) => handleOverwriteConfirm(v === "yes")}
          />
        </Box>
      </Box>
    );
  }

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
        <Text dimColor>Press enter to use default (./out), Esc to go back</Text>
      </Box>
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Step 5: API Key Setup
// ═════════════════════════════════════════════════════════════
function KeySetupStep({
  mode,
  selectedProviders,
  onComplete,
  onBack,
}: {
  mode: EvalMode;
  selectedProviders: string[];
  onComplete: (env: Record<string, string>) => void;
  onBack: () => void;
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
      const p = getProviderById(id, mode);
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
  }, [selectedProviders, mode]);

  const missingKeys = allKeys.filter((k) => !k.found);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [inputValue, setInputValue] = useState("");
  const [enteredKeys, setEnteredKeys] = useState<Set<string>>(new Set());
  const completedRef = useRef(false);

  useInput((_input, key) => {
    if (key.escape) {
      onBack();
    }
  });

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
        <Text dimColor>
          Enter to submit. Keys are stored in ~/.calibrate/credentials.json.
          Press Esc to go back.
        </Text>
      </Box>
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Step 6: Running Evaluations (max 2 providers in parallel)
// ═════════════════════════════════════════════════════════════
const MAX_PARALLEL_PROVIDERS = 2;
const BASE_PORT = 8765;

function RunStep({
  config,
  onComplete,
}: {
  config: EvalConfig;
  onComplete: () => void;
}) {
  const [states, setStates] = useState<Record<string, ProviderState>>(() => {
    const s: Record<string, ProviderState> = {};
    for (const p of config.providers) {
      s[p] = { status: "waiting", logs: [] };
    }
    return s;
  });
  const [phase, setPhase] = useState<"eval" | "done">("eval");
  const processRefs = useRef<Map<string, ChildProcess>>(new Map());
  const [runningCount, setRunningCount] = useState(0);
  const [nextProviderIdx, setNextProviderIdx] = useState(0);
  const usedPorts = useRef<Set<number>>(new Set());

  // Build spawn args for a provider eval
  function buildEvalArgs(provider: string, isLastProvider: boolean): string[] {
    const args = [
      ...config.calibrate.args,
      config.mode,
      "-p",
      provider,
      "-l",
      config.language,
      "-i",
      config.inputPath,
      "-o",
      config.outputDir,
    ];
    if (config.overwrite) {
      args.push("--overwrite");
    }
    // Generate leaderboard after the last provider eval
    if (isLastProvider) {
      args.push("--leaderboard");
    }
    return args;
  }

  // Start a provider evaluation
  const startProvider = async (provider: string, isLastProvider: boolean) => {
    // Find an available port
    let port = BASE_PORT;
    while (usedPorts.current.has(port)) {
      port++;
    }
    const availablePort = await findAvailablePort(port);
    if (availablePort) {
      usedPorts.current.add(availablePort);
    }

    setStates((prev) => ({
      ...prev,
      [provider]: { ...prev[provider]!, status: "running" },
    }));
    setRunningCount((c) => c + 1);

    const proc = spawn(
      config.calibrate.cmd,
      buildEvalArgs(provider, isLastProvider),
      {
        env: {
          ...process.env,
          ...config.envVars,
          PYTHONUNBUFFERED: "1", // Ensure Python output is not buffered
        },
        stdio: ["pipe", "pipe", "pipe"],
      }
    );

    processRefs.current.set(provider, proc);

    const onData = (data: Buffer) => {
      const lines = data
        .toString()
        .split(/[\r\n]+/)
        .filter((l) => l.trim());
      setStates((prev) => ({
        ...prev,
        [provider]: {
          ...prev[provider]!,
          logs: [...prev[provider]!.logs, ...lines].slice(-20),
        },
      }));
    };

    proc.stdout?.on("data", onData);
    proc.stderr?.on("data", onData);

    proc.on("error", () => {
      if (availablePort) usedPorts.current.delete(availablePort);
      setStates((prev) => ({
        ...prev,
        [provider]: { ...prev[provider]!, status: "error" },
      }));
      setRunningCount((c) => c - 1);
      processRefs.current.delete(provider);
    });

    proc.on("close", (code) => {
      if (availablePort) usedPorts.current.delete(availablePort);
      let metrics: ProviderState["metrics"] = undefined;
      if (code === 0) {
        try {
          const metricsPath = path.join(
            config.outputDir,
            provider,
            "metrics.json"
          );
          const raw = JSON.parse(fs.readFileSync(metricsPath, "utf-8"));
          if (config.mode === "tts") {
            metrics = {
              llm_judge_score: raw.llm_judge_score ?? 0,
              ttfb: raw.ttfb?.mean ?? raw.ttfb ?? 0,
            };
          } else {
            metrics = {
              wer: raw.wer ?? 0,
              string_similarity: raw.string_similarity ?? 0,
              llm_judge_score: raw.llm_judge_score ?? 0,
            };
          }
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
      setRunningCount((c) => c - 1);
      processRefs.current.delete(provider);
    });
  };

  // Effect to manage parallel provider execution
  useEffect(() => {
    if (phase !== "eval") return;

    // Check if all providers are done
    const completedCount = Object.values(states).filter(
      (s) => s.status === "done" || s.status === "error"
    ).length;

    if (completedCount >= config.providers.length) {
      setPhase("done");
      setTimeout(() => onComplete(), 500);
      return;
    }

    // Start more providers if we have capacity
    if (
      runningCount < MAX_PARALLEL_PROVIDERS &&
      nextProviderIdx < config.providers.length
    ) {
      const provider = config.providers[nextProviderIdx]!;
      const isLastProvider = nextProviderIdx === config.providers.length - 1;
      setNextProviderIdx((idx) => idx + 1);
      startProvider(provider, isLastProvider);
    }
  }, [phase, runningCount, nextProviderIdx, states]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      processRefs.current.forEach((proc) => proc.kill());
    };
  }, []);

  const completedCount = Object.values(states).filter(
    (s) => s.status === "done" || s.status === "error"
  ).length;

  // Get currently running providers for log display
  const runningProviders = config.providers.filter(
    (p) => states[p]?.status === "running"
  );

  // Inline metric summary
  function renderMetricSummary(state: ProviderState) {
    if (!state.metrics) return null;
    if (config.mode === "tts") {
      return (
        <Text dimColor>
          llm_judge: {state.metrics.llm_judge_score?.toFixed(2)}
          {"  "}ttfb: {state.metrics.ttfb?.toFixed(2)}s
        </Text>
      );
    } else {
      return (
        <Text dimColor>
          wer: {state.metrics.wer?.toFixed(2)}
          {"  "}similarity: {state.metrics.string_similarity?.toFixed(2)}
          {"  "}llm_judge: {state.metrics.llm_judge_score?.toFixed(2)}
        </Text>
      );
    }
  }

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          {getModeLabel(config.mode)} Evaluation
        </Text>
        <Text dimColor>
          {"  "}
          {completedCount}/{config.providers.length} providers
          {runningCount > 1 && ` (${runningCount} running in parallel)`}
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
              renderMetricSummary(state)
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

      {/* Log windows for running providers - side by side columns */}
      {phase === "eval" && runningProviders.length > 0 && (
        <Box flexDirection="row" marginTop={1}>
          {runningProviders.map((provider, idx) => (
            <Box
              key={provider}
              flexDirection="column"
              width="50%"
              marginRight={idx < runningProviders.length - 1 ? 1 : 0}
            >
              <Box>
                <Text dimColor>{"── "}</Text>
                <Text bold color="cyan">
                  {provider}
                </Text>
                <Text dimColor>
                  {" " + "\u2500".repeat(Math.max(0, 20 - provider.length))}
                </Text>
              </Box>
              <Box flexDirection="column" paddingLeft={1}>
                {(states[provider]?.logs || []).slice(-8).map((line, i) => (
                  <Text key={i} dimColor wrap="truncate">
                    {stripAnsi(line).slice(0, 45)}
                  </Text>
                ))}
              </Box>
            </Box>
          ))}
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
// Step 7: Leaderboard Display with Provider Details
// ═════════════════════════════════════════════════════════════
type ResultsView = "leaderboard" | "provider-detail";

interface ProviderResult {
  id: string;
  [key: string]: string | number | boolean;
}

function LeaderboardStep({ config }: { config: EvalConfig }) {
  const { exit } = useApp();
  const [view, setView] = useState<ResultsView>("leaderboard");
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [providerResults, setProviderResults] = useState<ProviderResult[]>([]);
  const [scrollOffset, setScrollOffset] = useState(0);
  const [selectedRowIdx, setSelectedRowIdx] = useState(0);
  const [playingAudio, setPlayingAudio] = useState<string | null>(null);
  const audioProcessRef = useRef<ChildProcess | null>(null);
  const MAX_VISIBLE_ROWS = 10;

  const [metrics, setMetrics] = useState<
    Array<{
      provider: string;
      [key: string]: string | number;
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

        if (config.mode === "tts") {
          results.push({
            provider,
            llm_judge_score: data.llm_judge_score ?? 0,
            ttfb: data.ttfb?.mean ?? data.ttfb ?? 0,
            count,
          });
        } else {
          results.push({
            provider,
            wer: data.wer ?? 0,
            string_similarity: data.string_similarity ?? 0,
            llm_judge_score: data.llm_judge_score ?? 0,
            count,
          });
        }
      } catch {
        // skip providers with no metrics
      }
    }
    setMetrics(results);
  }, []);

  // Parse CSV line handling quoted fields with commas
  const parseCSVLine = (line: string): string[] => {
    const result: string[] = [];
    let current = "";
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i]!;
      if (char === '"') {
        if (inQuotes && line[i + 1] === '"') {
          // Escaped quote
          current += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === "," && !inQuotes) {
        result.push(current.trim());
        current = "";
      } else {
        current += char;
      }
    }
    result.push(current.trim());
    return result;
  };

  // Load provider results when selected
  useEffect(() => {
    if (!selectedProvider) return;
    try {
      const resultsPath = path.join(
        config.outputDir,
        selectedProvider,
        "results.csv"
      );
      const csvContent = fs.readFileSync(resultsPath, "utf-8");
      const lines = csvContent.trim().split("\n");
      if (lines.length < 2) {
        setProviderResults([]);
        return;
      }
      const headers = parseCSVLine(lines[0]!);
      const rows: ProviderResult[] = [];
      for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]!);
        const row: ProviderResult = { id: "" };
        headers.forEach((h, idx) => {
          const val = values[idx] || "";
          // Try to parse as number for numeric fields
          if (
            ["wer", "string_similarity", "llm_judge_score", "ttfb"].includes(h)
          ) {
            const num = parseFloat(val);
            row[h] = isNaN(num) ? val : num;
          } else {
            row[h] = val;
          }
        });
        rows.push(row);
      }
      setProviderResults(rows);
      setScrollOffset(0);
      setSelectedRowIdx(0);
    } catch {
      setProviderResults([]);
    }
  }, [selectedProvider, config.outputDir]);

  // Play audio file for a given row ID
  const playAudio = (rowId: string) => {
    // Stop any currently playing audio
    if (audioProcessRef.current) {
      audioProcessRef.current.kill();
      audioProcessRef.current = null;
    }

    if (!selectedProvider) return;

    const audioPath = path.join(
      config.outputDir,
      selectedProvider,
      "audios",
      `${rowId}.wav`
    );

    if (!fs.existsSync(audioPath)) {
      return;
    }

    setPlayingAudio(rowId);

    // Use afplay on macOS, aplay on Linux
    const isLinux = process.platform === "linux";
    const cmd = isLinux ? "aplay" : "afplay";

    const proc = spawn(cmd, [audioPath], {
      stdio: ["ignore", "ignore", "ignore"],
    });

    audioProcessRef.current = proc;

    proc.on("close", () => {
      setPlayingAudio(null);
      audioProcessRef.current = null;
    });

    proc.on("error", () => {
      setPlayingAudio(null);
      audioProcessRef.current = null;
    });
  };

  // Stop audio playback
  const stopAudio = () => {
    if (audioProcessRef.current) {
      audioProcessRef.current.kill();
      audioProcessRef.current = null;
      setPlayingAudio(null);
    }
  };

  // Cleanup audio on unmount
  useEffect(() => {
    return () => {
      if (audioProcessRef.current) {
        audioProcessRef.current.kill();
      }
    };
  }, []);

  useInput((input, key) => {
    if (input === "q") {
      stopAudio();
      if (view === "provider-detail") {
        setView("leaderboard");
        setSelectedProvider(null);
        setScrollOffset(0);
        setSelectedRowIdx(0);
      } else {
        exit();
      }
    }
    if (key.escape && view === "provider-detail") {
      stopAudio();
      setView("leaderboard");
      setSelectedProvider(null);
      setScrollOffset(0);
      setSelectedRowIdx(0);
    }
    // Navigation and audio controls in TTS provider detail view
    if (view === "provider-detail" && config.mode === "tts") {
      if (key.upArrow) {
        setSelectedRowIdx((idx) => {
          const newIdx = Math.max(0, idx - 1);
          // Adjust scroll if needed
          if (newIdx < scrollOffset) {
            setScrollOffset(newIdx);
          }
          return newIdx;
        });
      }
      if (key.downArrow) {
        setSelectedRowIdx((idx) => {
          const newIdx = Math.min(providerResults.length - 1, idx + 1);
          // Adjust scroll if needed
          if (newIdx >= scrollOffset + MAX_VISIBLE_ROWS) {
            setScrollOffset(newIdx - MAX_VISIBLE_ROWS + 1);
          }
          return newIdx;
        });
      }
      // Play audio with Enter or 'p'
      if ((key.return || input === "p") && providerResults[selectedRowIdx]) {
        const rowId = String(providerResults[selectedRowIdx]!.id);
        if (playingAudio === rowId) {
          stopAudio();
        } else {
          playAudio(rowId);
        }
      }
      // Stop audio with 's'
      if (input === "s") {
        stopAudio();
      }
    }
    // Scroll in STT provider detail view (non-TTS keeps old behavior)
    if (view === "provider-detail" && config.mode !== "tts") {
      if (key.upArrow && scrollOffset > 0) {
        setScrollOffset((o) => o - 1);
      }
      if (
        key.downArrow &&
        scrollOffset < providerResults.length - MAX_VISIBLE_ROWS
      ) {
        setScrollOffset((o) => o + 1);
      }
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
  const leaderboardFile =
    config.mode === "tts" ? "tts_leaderboard.xlsx" : "stt_leaderboard.xlsx";

  // Provider Detail View
  if (view === "provider-detail" && selectedProvider) {
    const visibleRows = providerResults.slice(
      scrollOffset,
      scrollOffset + MAX_VISIBLE_ROWS
    );
    const truncate = (s: string, max: number) =>
      s.length > max ? s.slice(0, max - 1) + "…" : s;

    return (
      <Box flexDirection="column" padding={1}>
        <Box marginBottom={1}>
          <Text bold color="cyan">
            {selectedProvider} — Row-by-Row Results
          </Text>
          <Text dimColor> ({providerResults.length} rows)</Text>
        </Box>

        {providerResults.length === 0 ? (
          <Text color="yellow">No results found for this provider.</Text>
        ) : (
          <>
            {/* Results Table */}
            {config.mode === "tts" ? (
              <Box flexDirection="column">
                {/* Header */}
                <Box>
                  <Text bold> {" ".padEnd(6)}</Text>
                  <Text bold> | {"ID".padEnd(10)}</Text>
                  <Text bold> | {"Text".padEnd(28)}</Text>
                  <Text bold> | {"TTFB".padStart(8)}</Text>
                  <Text bold> | {"LLM Judge".padStart(10)}</Text>
                </Box>
                {/* Separator */}
                <Text dimColor>
                  {" " +
                    "-".repeat(6) +
                    "-+-" +
                    "-".repeat(10) +
                    "-+-" +
                    "-".repeat(28) +
                    "-+-" +
                    "-".repeat(8) +
                    "-+-" +
                    "-".repeat(10)}
                </Text>
                {/* Rows */}
                {visibleRows.map((r, idx) => {
                  const absoluteIdx = scrollOffset + idx;
                  const isSelected = absoluteIdx === selectedRowIdx;
                  const rowId = String(r.id || "");
                  const isPlaying = playingAudio === rowId;
                  const llmJudge =
                    r.llm_judge_score === true ||
                    r.llm_judge_score === "True" ||
                    r.llm_judge_score === 1
                      ? "Pass"
                      : r.llm_judge_score === false ||
                        r.llm_judge_score === "False" ||
                        r.llm_judge_score === 0
                      ? "Fail"
                      : String(r.llm_judge_score || "-");

                  return (
                    <Box key={idx}>
                      <Text
                        color={isSelected ? "cyan" : undefined}
                        bold={isSelected}
                      >
                        {isSelected ? " > " : "   "}
                      </Text>
                      <Text
                        color={
                          isPlaying ? "green" : isSelected ? "cyan" : undefined
                        }
                      >
                        {isPlaying ? "▶ Stop" : "  Play"}
                      </Text>
                      <Text color={isSelected ? "cyan" : undefined}>
                        {" | " + truncate(rowId, 10).padEnd(10)}
                      </Text>
                      <Text color={isSelected ? "cyan" : undefined}>
                        {" | " + truncate(String(r.text || ""), 28).padEnd(28)}
                      </Text>
                      <Text color={isSelected ? "cyan" : undefined}>
                        {" | " +
                          (typeof r.ttfb === "number"
                            ? r.ttfb.toFixed(2) + "s"
                            : "-"
                          ).padStart(8)}
                      </Text>
                      <Text color={isSelected ? "cyan" : undefined}>
                        {" | " + llmJudge.padStart(10)}
                      </Text>
                    </Box>
                  );
                })}
              </Box>
            ) : (
              <Table
                columns={[
                  { key: "id", label: "ID", width: 10 },
                  { key: "gt", label: "Ground Truth", width: 25 },
                  { key: "pred", label: "Prediction", width: 25 },
                  { key: "wer", label: "WER", width: 6, align: "right" },
                  {
                    key: "similarity",
                    label: "Sim",
                    width: 6,
                    align: "right",
                  },
                  {
                    key: "llm_judge",
                    label: "Judge",
                    width: 6,
                    align: "right",
                  },
                ]}
                data={visibleRows.map((r) => ({
                  id: truncate(String(r.id || ""), 10),
                  gt: truncate(String(r.gt || ""), 25),
                  pred: truncate(String(r.pred || ""), 25),
                  wer: typeof r.wer === "number" ? r.wer.toFixed(2) : "-",
                  similarity:
                    typeof r.string_similarity === "number"
                      ? r.string_similarity.toFixed(2)
                      : "-",
                  llm_judge:
                    r.llm_judge_score === true ||
                    r.llm_judge_score === "True" ||
                    r.llm_judge_score === 1
                      ? "Pass"
                      : r.llm_judge_score === false ||
                        r.llm_judge_score === "False" ||
                        r.llm_judge_score === 0
                      ? "Fail"
                      : String(r.llm_judge_score || "-"),
                }))}
              />
            )}

            {/* Scroll indicator */}
            {providerResults.length > MAX_VISIBLE_ROWS && (
              <Box marginTop={1}>
                <Text dimColor>
                  Showing {scrollOffset + 1}-
                  {Math.min(
                    scrollOffset + MAX_VISIBLE_ROWS,
                    providerResults.length
                  )}{" "}
                  of {providerResults.length}
                  {config.mode === "tts"
                    ? " | ↑↓ navigate, Enter/p play, s stop"
                    : " | Use ↑↓ to scroll"}
                </Text>
              </Box>
            )}

            {/* Audio controls hint for TTS */}
            {config.mode === "tts" &&
              providerResults.length <= MAX_VISIBLE_ROWS && (
                <Box marginTop={1}>
                  <Text dimColor>↑↓ navigate | </Text>
                  <Text color="yellow">Enter</Text>
                  <Text dimColor>/</Text>
                  <Text color="yellow">p</Text>
                  <Text dimColor> play audio | </Text>
                  <Text color="yellow">s</Text>
                  <Text dimColor> stop</Text>
                </Box>
              )}

            {/* LLM Judge Reasoning for visible rows */}
            <Box marginTop={1} flexDirection="column">
              <Text bold dimColor>
                LLM Judge Reasoning:
              </Text>
              {visibleRows.map((r, idx) => {
                const reasoning = String(r.llm_judge_reasoning || "");
                if (!reasoning || reasoning === "-") return null;
                const passed =
                  r.llm_judge_score === true ||
                  r.llm_judge_score === "True" ||
                  r.llm_judge_score === 1;
                return (
                  <Box key={idx} marginTop={1} flexDirection="column">
                    <Box>
                      <Text color={passed ? "green" : "red"}>
                        [{String(r.id || idx + 1)}]{" "}
                      </Text>
                      <Text color={passed ? "green" : "red"}>
                        {passed ? "Pass" : "Fail"}
                      </Text>
                    </Box>
                    <Box marginLeft={2}>
                      <Text wrap="wrap">{reasoning}</Text>
                    </Box>
                  </Box>
                );
              })}
            </Box>
          </>
        )}

        <Box marginTop={1}>
          <Text dimColor>
            {config.mode === "tts"
              ? "q/Esc back | ↑↓ navigate | Enter/p play | s stop"
              : "Press q or Esc to go back to leaderboard"}
          </Text>
        </Box>
      </Box>
    );
  }

  // Leaderboard View (default)
  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          {getModeLabel(config.mode)} Leaderboard
        </Text>
      </Box>

      {/* Comparison Table */}
      {config.mode === "tts" ? (
        <Table
          columns={[
            { key: "provider", label: "Provider", width: 14 },
            { key: "llm_judge", label: "LLM Judge", width: 10, align: "right" },
            { key: "ttfb", label: "TTFB (avg)", width: 11, align: "right" },
            { key: "count", label: "Count", width: 6, align: "right" },
          ]}
          data={metrics.map((m) => ({
            provider: m.provider as string,
            llm_judge: (m.llm_judge_score as number).toFixed(2),
            ttfb: (m.ttfb as number).toFixed(2) + "s",
            count: String(m.count),
          }))}
        />
      ) : (
        <Table
          columns={[
            { key: "provider", label: "Provider", width: 14 },
            { key: "wer", label: "WER", width: 8, align: "right" },
            {
              key: "similarity",
              label: "Similarity",
              width: 11,
              align: "right",
            },
            { key: "llm_judge", label: "LLM Judge", width: 10, align: "right" },
            { key: "count", label: "Count", width: 6, align: "right" },
          ]}
          data={metrics.map((m) => ({
            provider: m.provider as string,
            wer: (m.wer as number).toFixed(2),
            similarity: (m.string_similarity as number).toFixed(2),
            llm_judge: (m.llm_judge_score as number).toFixed(2),
            count: String(m.count),
          }))}
        />
      )}

      {/* Charts */}
      {config.mode === "tts" ? (
        <>
          {/* LLM Judge Score bar chart */}
          <Box marginTop={1} flexDirection="column">
            <Text bold>LLM Judge Score</Text>
            <BarChart
              data={metrics.map((m) => ({
                label: m.provider as string,
                value: m.llm_judge_score as number,
                color: "green",
              }))}
            />
          </Box>

          {/* TTFB bar chart */}
          <Box marginTop={1} flexDirection="column">
            <Box>
              <Text bold>TTFB </Text>
              <Text dimColor>(lower is better)</Text>
            </Box>
            <BarChart
              data={[...metrics]
                .sort((a, b) => (a.ttfb as number) - (b.ttfb as number))
                .map((m) => ({
                  label: m.provider as string,
                  value: m.ttfb as number,
                  color: "yellow",
                }))}
            />
          </Box>
        </>
      ) : (
        <>
          {/* WER bar chart */}
          <Box marginTop={1} flexDirection="column">
            <Box>
              <Text bold>Word Error Rate </Text>
              <Text dimColor>(lower is better)</Text>
            </Box>
            <BarChart
              data={[...metrics]
                .sort((a, b) => (a.wer as number) - (b.wer as number))
                .map((m) => ({
                  label: m.provider as string,
                  value: m.wer as number,
                  color: "yellow",
                }))}
            />
          </Box>

          {/* String Similarity bar chart */}
          <Box marginTop={1} flexDirection="column">
            <Text bold>String Similarity</Text>
            <BarChart
              data={metrics.map((m) => ({
                label: m.provider as string,
                value: m.string_similarity as number,
                color: "green",
              }))}
            />
          </Box>

          {/* LLM Judge Score bar chart */}
          <Box marginTop={1} flexDirection="column">
            <Text bold>LLM Judge Score</Text>
            <BarChart
              data={metrics.map((m) => ({
                label: m.provider as string,
                value: m.llm_judge_score as number,
                color: "green",
              }))}
            />
          </Box>
        </>
      )}

      {/* Provider selection to view details */}
      <Box marginTop={1} flexDirection="column">
        <Text dimColor>{"\u2500".repeat(50)}</Text>
        <Box marginTop={1}>
          <Text bold>View Provider Details</Text>
        </Box>
        <Box marginTop={1}>
          <SelectInput
            items={[
              ...config.providers.map((p) => ({
                label: `${p} — View row-by-row results`,
                value: p,
              })),
              { label: "Exit", value: "__exit__" },
            ]}
            onSelect={(v) => {
              if (v === "__exit__") {
                exit();
              } else {
                setSelectedProvider(v);
                setView("provider-detail");
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
          {config.mode === "tts" ? (
            <Box>
              <Text>{"  Audio:       "}</Text>
              <Text color="cyan">
                {resolvedOutputDir}/{"<provider>"}/audios/
              </Text>
            </Box>
          ) : null}
          <Box>
            <Text>{"  Results:     "}</Text>
            <Text color="cyan">
              {resolvedOutputDir}/{"<provider>"}/results.csv
            </Text>
          </Box>
          <Box>
            <Text>{"  Leaderboard: "}</Text>
            <Text color="cyan">
              {path.resolve(leaderboardDir)}/{leaderboardFile}
            </Text>
          </Box>
          <Box>
            <Text>{"  Charts:      "}</Text>
            <Text color="cyan">{path.resolve(leaderboardDir)}/</Text>
          </Box>
        </Box>
      </Box>
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Main Menu
// ═════════════════════════════════════════════════════════════
function MainMenu({ onSelect }: { onSelect: (mode: AppMode) => void }) {
  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Calibrate
        </Text>
        <Text bold> — Voice Agent Evaluation Toolkit</Text>
      </Box>
      <SelectInput
        items={[
          {
            label: "STT Evaluation        Benchmark speech-to-text providers",
            value: "stt",
          },
          {
            label: "TTS Evaluation        Benchmark text-to-speech providers",
            value: "tts",
          },
          {
            label: "LLM Tests             Test agent responses and tool calls",
            value: "llm",
          },
          {
            label: "Simulations           Run text or voice simulations",
            value: "simulations",
          },
        ]}
        onSelect={(v) => {
          onSelect(v as AppMode);
        }}
      />
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Eval App (STT/TTS — existing flow)
// ═════════════════════════════════════════════════════════════
function EvalApp({
  evalMode,
  onBack,
}: {
  evalMode: EvalMode;
  onBack?: () => void;
}) {
  const [step, setStep] = useState<Step | "init">("init");
  const [evalDone, setEvalDone] = useState(false);
  const [config, setConfig] = useState<EvalConfig>({
    mode: evalMode,
    providers: [],
    inputPath: "",
    language: "english",
    outputDir: "./out",
    overwrite: false,
    envVars: {},
    calibrate: { cmd: "calibrate", args: [] },
  });
  const [initError, setInitError] = useState("");

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

  // Show leaderboard step after evaluation completes
  if (evalDone) {
    return <LeaderboardStep config={config} />;
  }

  switch (step) {
    case "config-language":
      return (
        <ConfigLanguageStep
          mode={config.mode}
          onComplete={(lang) => {
            setConfig((c) => ({ ...c, language: lang }));
            setStep("select-providers");
          }}
          onBack={onBack}
        />
      );

    case "select-providers":
      return (
        <ProviderSelectStep
          mode={config.mode}
          language={config.language}
          onComplete={(providers) => {
            setConfig((c) => ({ ...c, providers }));
            setStep("config-input");
          }}
          onBack={() => setStep("config-language")}
        />
      );

    case "config-input":
      return (
        <ConfigInputStep
          mode={config.mode}
          onComplete={(inputPath) => {
            setConfig((c) => ({ ...c, inputPath }));
            setStep("config-output");
          }}
          onBack={() => setStep("select-providers")}
        />
      );

    case "config-output":
      return (
        <ConfigOutputStep
          providers={config.providers}
          onComplete={(dir, overwrite) => {
            setConfig((c) => ({ ...c, outputDir: dir, overwrite }));
            setStep("setup-keys");
          }}
          onBack={() => setStep("config-input")}
        />
      );

    case "setup-keys":
      return (
        <KeySetupStep
          mode={config.mode}
          selectedProviders={config.providers}
          onComplete={(envVars) => {
            setConfig((c) => ({ ...c, envVars }));
            setStep("running");
          }}
          onBack={() => setStep("config-output")}
        />
      );

    case "running":
      return <RunStep config={config} onComplete={() => setEvalDone(true)} />;

    default:
      return null;
  }
}

// ═════════════════════════════════════════════════════════════
// Main App — Routes to the appropriate flow
// ═════════════════════════════════════════════════════════════
export function App({ mode }: { mode: Mode }) {
  const [currentMode, setCurrentMode] = useState<AppMode>(mode);

  const goToMenu = () => setCurrentMode("menu");

  switch (currentMode) {
    case "menu":
      return <MainMenu onSelect={setCurrentMode} />;

    case "stt":
      return (
        <EvalApp
          evalMode="stt"
          onBack={mode === "menu" ? goToMenu : undefined}
        />
      );

    case "tts":
      return (
        <EvalApp
          evalMode="tts"
          onBack={mode === "menu" ? goToMenu : undefined}
        />
      );

    case "llm":
      return <LlmTestsApp onBack={goToMenu} />;

    case "simulations":
      return <SimulationsApp onBack={goToMenu} />;

    default:
      return <MainMenu onSelect={setCurrentMode} />;
  }
}
