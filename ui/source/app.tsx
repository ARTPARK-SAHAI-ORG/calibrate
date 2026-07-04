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
  // Optional path to a JSON config file with an ``evaluators`` list. When
  // unset, the backend falls back to its default evaluator (semantic_match
  // for STT, pronunciation for TTS).
  configFile?: string;
}

type Step =
  | "config-language"
  | "select-providers"
  | "config-input"
  | "config-output"
  | "config-file"
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

// Reserved metrics.json keys that look numeric but are not evaluator-
// derived (used by CSV-header scanners which don't have access to the
// nested metric values).
const RESERVED_METRIC_KEYS = new Set(["wer", "ttfb", "count", "provider"]);

type EvaluatorInfo = {
  type: "binary" | "rating";
  scale_min?: number;
  scale_max?: number;
};

// Each evaluator entry in metrics.json is a dict carrying a ``type`` field
// (``"binary"`` or ``"rating"``) alongside its ``mean``. That ``type`` key
// is the unambiguous marker for an evaluator vs other metrics like ``wer``
// (a plain number) or ``ttfb`` (a dict without ``type``).
function isEvaluatorEntry(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object") return false;
  const t = (value as Record<string, unknown>)["type"];
  return t === "binary" || t === "rating";
}

function evaluatorNamesFromMetrics(
  data: Record<string, unknown>
): string[] {
  return Object.keys(data).filter((k) => isEvaluatorEntry(data[k]));
}

function evaluatorScoresFromMetrics(
  data: Record<string, unknown>
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const name of evaluatorNamesFromMetrics(data)) {
    const entry = data[name] as Record<string, unknown>;
    const mean = entry["mean"];
    out[name] = typeof mean === "number" ? mean : Number(mean) || 0;
  }
  return out;
}

function evaluatorMetaFromMetrics(
  data: Record<string, unknown>
): Record<string, EvaluatorInfo> {
  const out: Record<string, EvaluatorInfo> = {};
  for (const name of evaluatorNamesFromMetrics(data)) {
    const entry = data[name] as Record<string, unknown>;
    const type = entry["type"] === "rating" ? "rating" : "binary";
    const meta: EvaluatorInfo = { type };
    if (type === "rating") {
      if (typeof entry["scale_min"] === "number")
        meta.scale_min = entry["scale_min"] as number;
      if (typeof entry["scale_max"] === "number")
        meta.scale_max = entry["scale_max"] as number;
    }
    out[name] = meta;
  }
  return out;
}

// Resolve the color for an evaluator score in row-by-row output.
// - binary: green for pass (true/1/"True"), red for fail (false/0/"False")
// - rating: red for scale_min, green for scale_max, yellow in between
function evaluatorScoreColor(
  score: unknown,
  meta: EvaluatorInfo | undefined
): string | undefined {
  if (
    meta?.type === "rating" &&
    typeof score === "number" &&
    typeof meta.scale_min === "number" &&
    typeof meta.scale_max === "number"
  ) {
    if (score <= meta.scale_min) return "red";
    if (score >= meta.scale_max) return "green";
    return "yellow";
  }
  const passed = score === true || score === "True" || score === 1;
  const failed = score === false || score === "False" || score === 0;
  return passed ? "green" : failed ? "red" : undefined;
}

// Evaluator columns in results.csv are paired: ``<name>`` carries the
// score and ``<name>_reasoning`` carries the judge's free-text reason.
// We discover evaluators by looking for headers that have a matching
// ``_reasoning`` companion — this is robust to arbitrary evaluator names.
function evaluatorNamesFromCsvHeaders(headers: string[]): string[] {
  const headerSet = new Set(headers);
  const names: string[] = [];
  for (const h of headers) {
    if (RESERVED_METRIC_KEYS.has(h)) continue;
    if (h.endsWith("_reasoning")) continue;
    if (headerSet.has(`${h}_reasoning`)) names.push(h);
  }
  return names;
}

function unionEvaluatorNames(
  metrics: Array<Record<string, string | number>>
): string[] {
  const seen = new Set<string>();
  const ordered: string[] = [];
  for (const m of metrics) {
    for (const k of Object.keys(m)) {
      if (RESERVED_METRIC_KEYS.has(k) || k === "provider" || k === "count") {
        continue;
      }
      if (typeof m[k] === "number" && !seen.has(k)) {
        seen.add(k);
        ordered.push(k);
      }
    }
  }
  return ordered;
}

// Format a per-evaluator score cell. Binary scores arrive as boolean-looking
// values (True/False/1/0); rating scores arrive as numbers and are rendered
// as integers (per-row rating scores are always integral by design).
function formatEvaluatorCell(val: unknown, meta?: EvaluatorInfo): string {
  if (meta?.type === "rating" && typeof val === "number") {
    return String(Math.round(val));
  }
  if (val === true || val === "True" || val === 1) return "Pass";
  if (val === false || val === "False" || val === 0) return "Fail";
  if (typeof val === "number") return val.toFixed(2);
  if (val === undefined || val === null || val === "") return "-";
  return String(val);
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
// Step 5: Optional config file (evaluators, judge model, prompts)
// ═════════════════════════════════════════════════════════════
function ConfigFileStep({
  mode,
  onComplete,
  onBack,
}: {
  mode: EvalMode;
  onComplete: (configPath: string | undefined) => void;
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
    if (!trimmed) {
      // Empty = skip; backend will use the default evaluator.
      onComplete(undefined);
      return;
    }
    if (!fs.existsSync(trimmed)) {
      setError(`File not found: ${trimmed}`);
      return;
    }
    try {
      const raw = fs.readFileSync(trimmed, "utf-8");
      const parsed = JSON.parse(raw);
      if (parsed === null || typeof parsed !== "object") {
        setError("Config file must contain a JSON object");
        return;
      }
    } catch (e) {
      setError(`Invalid JSON: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }
    onComplete(trimmed);
  };

  const docsUrl =
    mode === "tts"
      ? "https://calibrate.artpark.ai/docs/cli/text-to-speech"
      : "https://calibrate.artpark.ai/docs/cli/speech-to-text";

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Evaluator config (optional)
        </Text>
      </Box>
      <Box>
        <Text>Config JSON: </Text>
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
          <Text dimColor>
            Path to a JSON file with a top-level{" "}
            <Text color="yellow">evaluators</Text> list. Leave empty to use the
            default {mode === "tts" ? "pronunciation" : "semantic_match"}{" "}
            evaluator.
          </Text>
          <Text dimColor>
            See config format: <Text color="blue">{docsUrl}</Text>
          </Text>
        </Box>
      )}
      <Box marginTop={1}>
        <Text dimColor>Press enter to skip, Esc to go back</Text>
      </Box>
    </Box>
  );
}

// ═════════════════════════════════════════════════════════════
// Step 6: API Key Setup
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

    // Always need OPENAI_API_KEY for evaluators
    result.push({
      envVar: "OPENAI_API_KEY",
      label: "OpenAI (Evaluators)",
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
    if (config.configFile) {
      args.push("-c", config.configFile);
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
          const evaluatorScores = evaluatorScoresFromMetrics(raw);
          if (config.mode === "tts") {
            metrics = {
              ...evaluatorScores,
              ttfb: raw.ttfb?.mean ?? raw.ttfb ?? 0,
            };
          } else {
            metrics = {
              wer: raw.wer ?? 0,
              ...evaluatorScores,
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
    const m = state.metrics;
    const evalEntries = Object.entries(m).filter(
      ([k]) => !RESERVED_METRIC_KEYS.has(k)
    );
    const parts: string[] = [];
    if (config.mode === "tts") {
      for (const [name, val] of evalEntries) {
        parts.push(`${name}: ${val?.toFixed(2)}`);
      }
      if (typeof m.ttfb === "number") {
        parts.push(`ttfb: ${m.ttfb.toFixed(2)}s`);
      }
    } else {
      if (typeof m.wer === "number") {
        parts.push(`wer: ${m.wer.toFixed(2)}`);
      }
      for (const [name, val] of evalEntries) {
        parts.push(`${name}: ${val?.toFixed(2)}`);
      }
    }
    return <Text dimColor>{parts.join("  ")}</Text>;
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
  const [evaluatorMeta, setEvaluatorMeta] = useState<
    Record<string, EvaluatorInfo>
  >({});
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());

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

        const evaluatorScores = evaluatorScoresFromMetrics(data);
        if (config.mode === "tts") {
          results.push({
            provider,
            ...evaluatorScores,
            ttfb: data.ttfb?.mean ?? data.ttfb ?? 0,
            count,
          });
        } else {
          results.push({
            provider,
            wer: data.wer ?? 0,
            ...evaluatorScores,
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
      // Evaluator score columns sit next to a `<name>_reasoning` companion,
      // so we can detect them without relying on a suffix in the column name.
      const headerSet = new Set(headers);
      const evaluatorScoreColumns = new Set(
        headers.filter(
          (h) => !h.endsWith("_reasoning") && headerSet.has(`${h}_reasoning`)
        )
      );
      const rows: ProviderResult[] = [];
      for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]!);
        const row: ProviderResult = { id: "" };
        headers.forEach((h, idx) => {
          const val = values[idx] || "";
          // Numeric columns: wer, ttfb, and any per-evaluator score column.
          // Boolean-looking score values (True/False) stay as strings so the
          // row table can render Pass/Fail badges.
          const isNumericMetric =
            h === "wer" || h === "ttfb" || evaluatorScoreColumns.has(h);
          if (isNumericMetric) {
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
      setExpandedRows(new Set());
    } catch {
      setProviderResults([]);
    }

    try {
      const metricsPath = path.join(
        config.outputDir,
        selectedProvider,
        "metrics.json"
      );
      const data = JSON.parse(fs.readFileSync(metricsPath, "utf-8"));
      setEvaluatorMeta(evaluatorMetaFromMetrics(data));
    } catch {
      setEvaluatorMeta({});
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
    // Navigation and expand/collapse in STT provider detail view
    if (view === "provider-detail" && config.mode !== "tts") {
      if (key.upArrow) {
        setSelectedRowIdx((idx) => {
          const newIdx = Math.max(0, idx - 1);
          if (newIdx < scrollOffset) {
            setScrollOffset(newIdx);
          }
          return newIdx;
        });
      }
      if (key.downArrow) {
        setSelectedRowIdx((idx) => {
          const newIdx = Math.min(providerResults.length - 1, idx + 1);
          if (newIdx >= scrollOffset + MAX_VISIBLE_ROWS) {
            setScrollOffset(newIdx - MAX_VISIBLE_ROWS + 1);
          }
          return newIdx;
        });
      }
      if (key.return || input === " ") {
        setExpandedRows((prev) => {
          const next = new Set(prev);
          if (next.has(selectedRowIdx)) {
            next.delete(selectedRowIdx);
          } else {
            next.add(selectedRowIdx);
          }
          return next;
        });
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
    // Discover per-evaluator columns from results.csv headers — each
    // evaluator has a paired ``<name>`` score and ``<name>_reasoning`` column.
    const detailEvaluatorNames =
      providerResults.length > 0
        ? evaluatorNamesFromCsvHeaders(Object.keys(providerResults[0]!))
        : [];

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
                  {detailEvaluatorNames.map((name) => (
                    <Text key={name} bold>
                      {" | " + truncate(name, 10).padStart(10)}
                    </Text>
                  ))}
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
                    detailEvaluatorNames
                      .map(() => "-+-" + "-".repeat(10))
                      .join("")}
                </Text>
                {/* Rows */}
                {visibleRows.map((r, idx) => {
                  const absoluteIdx = scrollOffset + idx;
                  const isSelected = absoluteIdx === selectedRowIdx;
                  const rowId = String(r.id || "");
                  const isPlaying = playingAudio === rowId;

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
                      {detailEvaluatorNames.map((name) => (
                        <Text
                          key={name}
                          color={isSelected ? "cyan" : undefined}
                        >
                          {" | " +
                            formatEvaluatorCell(
                              r[name],
                              evaluatorMeta[name]
                            ).padStart(10)}
                        </Text>
                      ))}
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
                  ...detailEvaluatorNames.map((name) => ({
                    key: name,
                    label: name.length > 10 ? name.slice(0, 10) : name,
                    width: Math.max(6, Math.min(name.length, 10)),
                    align: "right" as const,
                  })),
                ]}
                data={visibleRows.map((r) => {
                  const row: Record<string, string> = {
                    id: truncate(String(r.id || ""), 10),
                    gt: truncate(String(r.gt || ""), 25),
                    pred: truncate(String(r.pred || ""), 25),
                    wer: typeof r.wer === "number" ? r.wer.toFixed(2) : "-",
                  };
                  for (const name of detailEvaluatorNames) {
                    row[name] = formatEvaluatorCell(
                      r[name],
                      evaluatorMeta[name]
                    );
                  }
                  return row;
                })}
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
                    : " | ↑↓ navigate, Enter to expand/collapse"}
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

            {/* Per-evaluator reasoning for visible rows (TTS) */}
            {config.mode === "tts" && detailEvaluatorNames.length > 0 && (
              <Box marginTop={1} flexDirection="column">
                <Text bold dimColor>
                  Evaluator Reasoning:
                </Text>
                {visibleRows.map((r, idx) => {
                  type ReasoningBlock = {
                    name: string;
                    reasoning: string;
                    color: string | undefined;
                    score: string | number | boolean;
                  };
                  const blocks: ReasoningBlock[] = [];
                  for (const name of detailEvaluatorNames) {
                    const reasoning = String(r[`${name}_reasoning`] || "");
                    if (!reasoning || reasoning === "-") continue;
                    const score = r[name] ?? "-";
                    const color = evaluatorScoreColor(
                      score,
                      evaluatorMeta[name]
                    );
                    blocks.push({ name, reasoning, color, score });
                  }
                  if (blocks.length === 0) return null;
                  return (
                    <Box key={idx} marginTop={1} flexDirection="column">
                      <Text dimColor>[{String(r.id || idx + 1)}]</Text>
                      {blocks.map((b) => (
                        <Box key={b.name} marginLeft={2} flexDirection="column">
                          <Box>
                            <Text color={b.color}>
                              {b.name}:{" "}
                              {formatEvaluatorCell(
                                b.score,
                                evaluatorMeta[b.name]
                              )}
                            </Text>
                          </Box>
                          <Box marginLeft={2}>
                            <Text wrap="wrap">{b.reasoning}</Text>
                          </Box>
                        </Box>
                      ))}
                    </Box>
                  );
                })}
              </Box>
            )}

            {/* Per-row details for STT — full GT/Pred always shown,
                evaluator reasoning toggled with Enter on the selected row. */}
            {config.mode !== "tts" && (
              <Box marginTop={1} flexDirection="column">
                <Text bold dimColor>
                  Row Details:
                </Text>
                {visibleRows.map((r, idx) => {
                  const absoluteIdx = scrollOffset + idx;
                  const isSelected = absoluteIdx === selectedRowIdx;
                  const isExpanded = expandedRows.has(absoluteIdx);

                  type ReasoningBlock = {
                    name: string;
                    reasoning: string;
                    color: string | undefined;
                    score: string | number | boolean;
                  };
                  const blocks: ReasoningBlock[] = [];
                  for (const name of detailEvaluatorNames) {
                    const reasoning = String(r[`${name}_reasoning`] || "");
                    if (!reasoning || reasoning === "-") continue;
                    const score = r[name] ?? "-";
                    const color = evaluatorScoreColor(
                      score,
                      evaluatorMeta[name]
                    );
                    blocks.push({ name, reasoning, color, score });
                  }
                  const hasBlocks = blocks.length > 0;

                  return (
                    <Box
                      key={absoluteIdx}
                      marginTop={1}
                      flexDirection="column"
                    >
                      <Box>
                        <Text
                          color={isSelected ? "cyan" : undefined}
                          bold={isSelected}
                        >
                          {(isSelected ? "> " : "  ") +
                            `[${String(r.id || idx + 1)}]`}
                        </Text>
                        {hasBlocks && (
                          <Text dimColor> {isExpanded ? "▼" : "▶"}</Text>
                        )}
                      </Box>
                      <Box marginLeft={4} flexDirection="column">
                        <Box>
                          <Text bold dimColor>
                            GT:{" "}
                          </Text>
                          <Text wrap="wrap">{String(r.gt || "")}</Text>
                        </Box>
                        <Box>
                          <Text bold dimColor>
                            Pred:{" "}
                          </Text>
                          <Text wrap="wrap">{String(r.pred || "")}</Text>
                        </Box>
                        {isExpanded && hasBlocks && (
                          <Box marginTop={1} flexDirection="column">
                            {blocks.map((b) => (
                              <Box
                                key={b.name}
                                flexDirection="column"
                                marginTop={1}
                              >
                                <Box>
                                  <Text color={b.color}>
                                    {b.name}:{" "}
                                    {formatEvaluatorCell(
                                      b.score,
                                      evaluatorMeta[b.name]
                                    )}
                                  </Text>
                                </Box>
                                <Box marginLeft={2}>
                                  <Text wrap="wrap">{b.reasoning}</Text>
                                </Box>
                              </Box>
                            ))}
                          </Box>
                        )}
                      </Box>
                    </Box>
                  );
                })}
              </Box>
            )}
          </>
        )}

        <Box marginTop={1}>
          <Text dimColor>
            {config.mode === "tts"
              ? "q/Esc back | ↑↓ navigate | Enter/p play | s stop"
              : "q/Esc back | ↑↓ navigate | Enter to expand/collapse"}
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
      {(() => {
        const lbEvaluatorNames = unionEvaluatorNames(metrics);
        const evaluatorColumns = lbEvaluatorNames.map((name) => ({
          key: name,
          label: name.length > 12 ? name.slice(0, 12) : name,
          width: Math.max(8, Math.min(name.length, 12)),
          align: "right" as const,
        }));
        const evaluatorCells = (m: (typeof metrics)[number]) => {
          const out: Record<string, string> = {};
          for (const name of lbEvaluatorNames) {
            const v = m[name];
            out[name] = typeof v === "number" ? v.toFixed(2) : "-";
          }
          return out;
        };
        return config.mode === "tts" ? (
          <Table
            columns={[
              { key: "provider", label: "Provider", width: 14 },
              ...evaluatorColumns,
              { key: "ttfb", label: "TTFB (avg)", width: 11, align: "right" },
              { key: "count", label: "Count", width: 6, align: "right" },
            ]}
            data={metrics.map((m) => ({
              provider: m.provider as string,
              ...evaluatorCells(m),
              ttfb:
                typeof m.ttfb === "number" ? m.ttfb.toFixed(2) + "s" : "-",
              count: String(m.count),
            }))}
          />
        ) : (
          <Table
            columns={[
              { key: "provider", label: "Provider", width: 14 },
              { key: "wer", label: "WER", width: 8, align: "right" },
              ...evaluatorColumns,
              { key: "count", label: "Count", width: 6, align: "right" },
            ]}
            data={metrics.map((m) => ({
              provider: m.provider as string,
              wer: typeof m.wer === "number" ? m.wer.toFixed(2) : "-",
              ...evaluatorCells(m),
              count: String(m.count),
            }))}
          />
        );
      })()}

      {/* Charts */}
      {(() => {
        const chartEvaluatorNames = unionEvaluatorNames(metrics);
        const evaluatorCharts = chartEvaluatorNames.map((name) => (
          <Box key={name} marginTop={1} flexDirection="column">
            <Text bold>{name}</Text>
            <BarChart
              data={metrics.map((m) => ({
                label: m.provider as string,
                value: typeof m[name] === "number" ? (m[name] as number) : 0,
                color: "green",
              }))}
            />
          </Box>
        ));
        return config.mode === "tts" ? (
          <>
            {evaluatorCharts}

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

            {evaluatorCharts}
          </>
        );
      })()}

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
    calibrate: { cmd: "calibrate-agent", args: [] },
  });
  const [initError, setInitError] = useState("");

  useEffect(() => {
    const result = findCalibrateBin();
    if (result) {
      setConfig((c) => ({ ...c, calibrate: result }));
      setStep("config-language");
    } else {
      setInitError(
        "calibrate-agent CLI not found. Install with: pip install -e . (from project root)"
      );
    }
  }, []);

  if (step === "init" && !initError) {
    return (
      <Box padding={1}>
        <Spinner label="Checking calibrate-agent CLI..." />
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
            setStep("config-file");
          }}
          onBack={() => setStep("config-input")}
        />
      );

    case "config-file":
      return (
        <ConfigFileStep
          mode={config.mode}
          onComplete={(configFile) => {
            setConfig((c) => ({ ...c, configFile }));
            setStep("setup-keys");
          }}
          onBack={() => setStep("config-output")}
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
          onBack={() => setStep("config-file")}
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
