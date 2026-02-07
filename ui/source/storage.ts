import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';

const STORE_DIR = path.join(os.homedir(), '.calibrate');

function ensureDir() {
  fs.mkdirSync(STORE_DIR, { recursive: true, mode: 0o700 });
}

function loadFile<T>(filename: string): T[] {
  try {
    const raw = fs.readFileSync(path.join(STORE_DIR, filename), 'utf-8');
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

function saveFile<T>(filename: string, data: T[]) {
  ensureDir();
  fs.writeFileSync(
    path.join(STORE_DIR, filename),
    JSON.stringify(data, null, 2),
    { mode: 0o600 },
  );
}

function genId(): string {
  return crypto.randomUUID().slice(0, 8);
}

// ─── Types ───────────────────────────────────────────────────

export interface StoredTool {
  id: string;
  name: string;
  type: 'structured_output' | 'webhook';
  description: string;
  parameters: Array<{
    id: string;
    type: string;
    description: string;
    required: boolean;
    items?: { type: string };
  }>;
  webhook?: {
    method: string;
    url: string;
    timeout: number;
    headers: Array<{ name: string; value: string }>;
    queryParameters: Array<{
      id: string;
      type: string;
      description: string;
      required: boolean;
    }>;
    body: {
      description: string;
      parameters: Array<{
        id: string;
        type: string;
        description: string;
        required: boolean;
      }>;
    };
  };
}

export interface StoredAgent {
  id: string;
  name: string;
  system_prompt: string;
  tools: StoredTool[];
  settings: {
    agent_speaks_first: boolean;
    max_turns: number;
  };
}

export interface StoredPersona {
  id: string;
  name: string;
  characteristics: string;
  gender: string;
  language: string;
  interruption_sensitivity: string;
}

export interface StoredScenario {
  id: string;
  name: string;
  description: string;
}

export interface StoredMetric {
  id: string;
  name: string;
  description: string;
}

// ─── CRUD ────────────────────────────────────────────────────

const FILES = {
  agents: 'agents.json',
  tools: 'tools.json',
  personas: 'personas.json',
  scenarios: 'scenarios.json',
  metrics: 'metrics.json',
} as const;

// Agents
export function listAgents(): StoredAgent[] { return loadFile(FILES.agents); }
export function addAgent(a: Omit<StoredAgent, 'id'>): StoredAgent {
  const items = listAgents();
  const item = { ...a, id: genId() };
  items.push(item);
  saveFile(FILES.agents, items);
  return item;
}
export function updateAgent(id: string, updates: Partial<Omit<StoredAgent, 'id'>>) {
  const items = listAgents();
  const idx = items.findIndex(a => a.id === id);
  if (idx >= 0) { items[idx] = { ...items[idx]!, ...updates, id }; saveFile(FILES.agents, items); }
}
export function removeAgent(id: string) {
  saveFile(FILES.agents, listAgents().filter(a => a.id !== id));
}

// Tools
export function listTools(): StoredTool[] { return loadFile(FILES.tools); }
export function addTool(t: Omit<StoredTool, 'id'>): StoredTool {
  const items = listTools();
  const item = { ...t, id: genId() };
  items.push(item);
  saveFile(FILES.tools, items);
  return item;
}
export function updateTool(id: string, updates: Partial<Omit<StoredTool, 'id'>>) {
  const items = listTools();
  const idx = items.findIndex(t => t.id === id);
  if (idx >= 0) { items[idx] = { ...items[idx]!, ...updates, id }; saveFile(FILES.tools, items); }
}
export function removeTool(id: string) {
  saveFile(FILES.tools, listTools().filter(t => t.id !== id));
}

// Personas
export function listPersonas(): StoredPersona[] { return loadFile(FILES.personas); }
export function addPersona(p: Omit<StoredPersona, 'id'>): StoredPersona {
  const items = listPersonas();
  const item = { ...p, id: genId() };
  items.push(item);
  saveFile(FILES.personas, items);
  return item;
}
export function updatePersona(id: string, updates: Partial<Omit<StoredPersona, 'id'>>) {
  const items = listPersonas();
  const idx = items.findIndex(p => p.id === id);
  if (idx >= 0) { items[idx] = { ...items[idx]!, ...updates, id }; saveFile(FILES.personas, items); }
}
export function removePersona(id: string) {
  saveFile(FILES.personas, listPersonas().filter(p => p.id !== id));
}

// Scenarios
export function listScenarios(): StoredScenario[] { return loadFile(FILES.scenarios); }
export function addScenario(s: Omit<StoredScenario, 'id'>): StoredScenario {
  const items = listScenarios();
  const item = { ...s, id: genId() };
  items.push(item);
  saveFile(FILES.scenarios, items);
  return item;
}
export function updateScenario(id: string, updates: Partial<Omit<StoredScenario, 'id'>>) {
  const items = listScenarios();
  const idx = items.findIndex(s => s.id === id);
  if (idx >= 0) { items[idx] = { ...items[idx]!, ...updates, id }; saveFile(FILES.scenarios, items); }
}
export function removeScenario(id: string) {
  saveFile(FILES.scenarios, listScenarios().filter(s => s.id !== id));
}

// Metrics (evaluation criteria)
export function listMetrics(): StoredMetric[] { return loadFile(FILES.metrics); }
export function addMetric(m: Omit<StoredMetric, 'id'>): StoredMetric {
  const items = listMetrics();
  const item = { ...m, id: genId() };
  items.push(item);
  saveFile(FILES.metrics, items);
  return item;
}
export function updateMetric(id: string, updates: Partial<Omit<StoredMetric, 'id'>>) {
  const items = listMetrics();
  const idx = items.findIndex(m => m.id === id);
  if (idx >= 0) { items[idx] = { ...items[idx]!, ...updates, id }; saveFile(FILES.metrics, items); }
}
export function removeMetric(id: string) {
  saveFile(FILES.metrics, listMetrics().filter(m => m.id !== id));
}

// ─── Config builders ─────────────────────────────────────────

/** Build a tool definition compatible with calibrate config JSON */
export function toolToConfig(t: StoredTool): Record<string, any> {
  const base: Record<string, any> = {
    type: t.type === 'webhook' ? 'webhook' : 'structured_output',
    name: t.name,
    description: t.description,
    parameters: t.parameters,
  };
  if (t.type === 'webhook' && t.webhook) {
    base.webhook = t.webhook;
  }
  return base;
}

/** Build LLM tests config JSON */
export function buildTestsConfig(
  agent: StoredAgent,
  testCases: any[],
): Record<string, any> {
  return {
    system_prompt: agent.system_prompt,
    tools: agent.tools.map(toolToConfig),
    test_cases: testCases,
  };
}

/** Build LLM simulation config JSON */
export function buildTextSimConfig(
  agent: StoredAgent,
  personas: StoredPersona[],
  scenarios: StoredScenario[],
  metrics: StoredMetric[],
): Record<string, any> {
  return {
    system_prompt: agent.system_prompt,
    tools: agent.tools.map(toolToConfig),
    personas: personas.map(p => ({
      characteristics: p.characteristics,
      gender: p.gender,
      language: p.language,
    })),
    scenarios: scenarios.map(s => ({ description: s.description })),
    evaluation_criteria: metrics.map(m => ({
      name: m.name,
      description: m.description,
    })),
    settings: agent.settings,
  };
}

/** Build voice agent simulation config JSON */
export function buildVoiceSimConfig(
  agent: StoredAgent,
  personas: StoredPersona[],
  scenarios: StoredScenario[],
  metrics: StoredMetric[],
  stt: { provider: string },
  tts: { provider: string },
  llm: { provider: string; model: string },
): Record<string, any> {
  return {
    system_prompt: agent.system_prompt,
    tools: agent.tools.map(toolToConfig),
    personas: personas.map(p => ({
      characteristics: p.characteristics,
      gender: p.gender,
      language: p.language,
      interruption_sensitivity: p.interruption_sensitivity,
    })),
    scenarios: scenarios.map(s => ({ description: s.description })),
    evaluation_criteria: metrics.map(m => ({
      name: m.name,
      description: m.description,
    })),
    stt,
    tts,
    llm,
    settings: agent.settings,
  };
}
