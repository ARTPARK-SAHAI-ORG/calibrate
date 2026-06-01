import { type CalibrateCmd } from "./shared.js";

// ─── Step Types ───────────────────────────────────────────────
export type LlmStep =
  | "init"
  | "config-path"
  | "provider"
  | "enter-model"
  | "model-confirm"
  | "agent-mode"
  | "agent-model-entry"
  | "agent-model-confirm"
  | "agent-verify"
  | "output-dir"
  | "output-dir-confirm"
  | "api-keys"
  | "running"
  | "leaderboard";

// ─── Interfaces ───────────────────────────────────────────────

// Per-evaluator aggregate parsed from a model's metrics.json `criteria` block.
// Binary evaluators contribute pass_rate; rating evaluators contribute mean.
// `display` is the formatted string ready for table/inline display.
export type EvaluatorAggregate = {
  type: "binary" | "rating";
  display: string; // e.g. "80.0%" (binary) or "4.20/5" (rating: mean/scale_max)
  sortValue: number;
};

export interface ModelState {
  status: "waiting" | "running" | "done" | "error";
  logs: string[];
  metrics?: {
    passed?: number;
    failed?: number;
    total?: number;
    evaluators?: Record<string, EvaluatorAggregate>;
  };
}

export interface HistoryMessage {
  role: string;
  content: string;
}

export interface ToolCall {
  tool: string;
  arguments: Record<string, unknown>;
  // Optional result the tool returned when an agent connection executed it.
  // Display-only — never affects evaluation. Internal LLM runs never set it.
  output?: unknown;
}

// Per-evaluator result keyed by evaluator name. Binary evaluators have a
// ``match`` boolean; rating evaluators have a numeric ``score``. Both have
// ``reasoning``.
export type JudgeEvaluatorResult = {
  match?: boolean;
  score?: number;
  reasoning?: string;
};

export interface TestResult {
  id: string;
  history: HistoryMessage[];
  evaluationType: string;
  evaluationCriteria: string;
  actualOutput: string;
  passed: boolean;
  reasoning: string;
  judgeResults?: Record<string, JudgeEvaluatorResult>;
}

export interface LlmConfig {
  configPath: string;
  models: string[];
  provider: string;
  outputDir: string;
  overwrite: boolean;
  envVars: Record<string, string>;
  calibrate: CalibrateCmd;
  agentUrl: string;
  agentHeaders: Record<string, string>;
  agentBenchmark: boolean;
  agentModels: string[];
}

// ─── Constants ────────────────────────────────────────────────
export const MAX_PARALLEL_MODELS = 2;

export const OPENAI_MODEL_EXAMPLES = [
  "gpt-4.1",
  "gpt-4.1-mini",
  "gpt-4o",
  "gpt-4o-mini",
  "o1",
  "o1-mini",
  "o3-mini",
];

export const OPENROUTER_MODEL_EXAMPLES = [
  "openai/gpt-4.1",
  "anthropic/claude-sonnet-4",
  "google/gemini-2.0-flash-001",
];
