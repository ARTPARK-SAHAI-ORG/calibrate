import React, { useState, useMemo } from "react";
import { Box, Text, useInput } from "ink";
import { SelectInput, TextInput, MultiSelect, TextArea } from "./components.js";
import {
  listAgents,
  addAgent,
  updateAgent,
  removeAgent,
  listTools,
  addTool,
  updateTool,
  removeTool,
  listPersonas,
  addPersona,
  updatePersona,
  removePersona,
  listScenarios,
  addScenario,
  updateScenario,
  removeScenario,
  listMetrics,
  addMetric,
  updateMetric,
  removeMetric,
  type StoredAgent,
  type StoredTool,
  type StoredPersona,
  type StoredScenario,
  type StoredMetric,
} from "./storage.js";

// ═══════════════════════════════════════════════════════════════
// Tool Creation
// ═══════════════════════════════════════════════════════════════
type ToolStep =
  | "type"
  | "name"
  | "description"
  | "param-name"
  | "param-type"
  | "param-desc"
  | "param-required"
  | "param-more"
  | "webhook-method"
  | "webhook-url";

const PARAM_TYPES = [
  "string",
  "integer",
  "number",
  "boolean",
  "array",
  "object",
];

export function CreateToolFlow({
  onComplete,
  onCancel,
}: {
  onComplete: (tool: StoredTool) => void;
  onCancel: () => void;
}) {
  const [step, setStep] = useState<ToolStep>("type");
  const [toolType, setToolType] = useState<"structured_output" | "webhook">(
    "structured_output"
  );
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [parameters, setParameters] = useState<StoredTool["parameters"]>([]);
  const [curParam, setCurParam] = useState({
    id: "",
    type: "string",
    description: "",
    required: true,
  });
  const [webhookMethod, setWebhookMethod] = useState("POST");
  const [webhookUrl, setWebhookUrl] = useState("");
  const [nameInput, setNameInput] = useState("");
  const [descInput, setDescInput] = useState("");
  const [paramNameInput, setParamNameInput] = useState("");
  const [paramDescInput, setParamDescInput] = useState("");
  const [urlInput, setUrlInput] = useState("");

  function saveTool() {
    const tool = addTool({
      name,
      type: toolType,
      description,
      parameters,
      ...(toolType === "webhook"
        ? {
            webhook: {
              method: webhookMethod,
              url: webhookUrl,
              timeout: 20,
              headers: [],
              queryParameters: [],
              body: { description: "", parameters: [] },
            },
          }
        : {}),
    });
    onComplete(tool);
  }

  const header = (
    <Box marginBottom={1}>
      <Text bold color="cyan">
        Create Tool
      </Text>
    </Box>
  );

  switch (step) {
    case "type":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text>Select tool type:</Text>
          <Text dimColor>
            Structured Output tools return data directly; Webhook tools call an
            external HTTP endpoint.
          </Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "Structured Output tool", value: "structured_output" },
                { label: "Webhook tool", value: "webhook" },
                { label: "Cancel", value: "_cancel" },
              ]}
              onSelect={(v) => {
                if (v === "_cancel") {
                  onCancel();
                  return;
                }
                setToolType(v as any);
                setStep("name");
              }}
            />
          </Box>
        </Box>
      );

    case "name":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            A unique identifier your agent will use to call this tool (use
            snake_case).
          </Text>
          <Box>
            <Text>Tool name: </Text>
            <TextInput
              value={nameInput}
              onChange={setNameInput}
              onSubmit={(v) => {
                if (v.trim()) {
                  setName(v.trim());
                  setStep("description");
                }
              }}
              placeholder="e.g. book_flight"
            />
          </Box>
        </Box>
      );

    case "description":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            Explain what this tool does so the LLM knows when to invoke it.
          </Text>
          <Box>
            <Text>Description: </Text>
            <TextInput
              value={descInput}
              onChange={setDescInput}
              onSubmit={(v) => {
                if (v.trim()) {
                  setDescription(v.trim());
                  setStep("param-name");
                }
              }}
              placeholder="What does this tool do?"
            />
          </Box>
        </Box>
      );

    case "param-name":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            Parameters ({parameters.length} added) — define the inputs your tool
            accepts.
          </Text>
          <Box marginTop={1}>
            <Text>Parameter name: </Text>
            <TextInput
              value={paramNameInput}
              onChange={setParamNameInput}
              onSubmit={(v) => {
                if (v.trim()) {
                  setCurParam((p) => ({ ...p, id: v.trim() }));
                  setParamNameInput("");
                  setStep("param-type");
                }
              }}
              placeholder="e.g. destination"
            />
          </Box>
        </Box>
      );

    case "param-type":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            Choose the data type for this parameter (e.g. string for text,
            integer for whole numbers).
          </Text>
          <Text>Type for "{curParam.id}":</Text>
          <Box marginTop={1}>
            <SelectInput
              items={PARAM_TYPES.map((t) => ({ label: t, value: t }))}
              onSelect={(v) => {
                setCurParam((p) => ({ ...p, type: v }));
                setStep("param-desc");
              }}
            />
          </Box>
        </Box>
      );

    case "param-desc":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            Help the LLM understand what value to pass for this parameter.
          </Text>
          <Box>
            <Text>Description for "{curParam.id}": </Text>
            <TextInput
              value={paramDescInput}
              onChange={setParamDescInput}
              onSubmit={(v) => {
                setCurParam((p) => ({ ...p, description: v.trim() }));
                setParamDescInput("");
                setStep("param-required");
              }}
              placeholder="Describe this parameter"
            />
          </Box>
        </Box>
      );

    case "param-required":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            Required parameters must always be provided when the tool is called.
          </Text>
          <Text>Is "{curParam.id}" required?</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "Yes", value: "yes" },
                { label: "No", value: "no" },
              ]}
              onSelect={(v) => {
                const param = { ...curParam, required: v === "yes" };
                setParameters((prev) => [...prev, param]);
                setCurParam({
                  id: "",
                  type: "string",
                  description: "",
                  required: true,
                });
                setStep("param-more");
              }}
            />
          </Box>
        </Box>
      );

    case "param-more":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>{parameters.length} parameter(s) added</Text>
          <Text>Add another parameter?</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "Yes, add another", value: "yes" },
                { label: "No, done with parameters", value: "no" },
              ]}
              onSelect={(v) => {
                if (v === "yes") {
                  setStep("param-name");
                } else if (toolType === "webhook") {
                  setStep("webhook-method");
                } else {
                  saveTool();
                }
              }}
            />
          </Box>
        </Box>
      );

    case "webhook-method":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            The HTTP method used when calling your webhook endpoint.
          </Text>
          <Text>HTTP Method:</Text>
          <Box marginTop={1}>
            <SelectInput
              items={["GET", "POST", "PUT", "PATCH", "DELETE"].map((m) => ({
                label: m,
                value: m,
              }))}
              onSelect={(v) => {
                setWebhookMethod(v);
                setStep("webhook-url");
              }}
            />
          </Box>
        </Box>
      );

    case "webhook-url":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            The endpoint URL that will be called when your agent invokes this
            tool.
          </Text>
          <Box>
            <Text>URL: </Text>
            <TextInput
              value={urlInput}
              onChange={setUrlInput}
              onSubmit={(v) => {
                if (v.trim()) {
                  setWebhookUrl(v.trim());
                  saveTool();
                }
              }}
              placeholder="https://api.example.com/endpoint"
            />
          </Box>
        </Box>
      );

    default:
      return null;
  }
}

// ═══════════════════════════════════════════════════════════════
// Persona Creation
// ═══════════════════════════════════════════════════════════════
type PersonaStep =
  | "name"
  | "characteristics"
  | "gender"
  | "language"
  | "interruption";

export function CreatePersonaFlow({
  onComplete,
  onCancel,
}: {
  onComplete: (persona: StoredPersona) => void;
  onCancel: () => void;
}) {
  const [step, setStep] = useState<PersonaStep>("name");
  const [name, setNameVal] = useState("");
  const [characteristics, setChar] = useState("");
  const [gender, setGender] = useState("");
  const [language, setLang] = useState("");
  const [nameInput, setNameInput] = useState("");

  const header = (
    <Box marginBottom={1}>
      <Text bold color="cyan">
        Create Persona
      </Text>
    </Box>
  );

  switch (step) {
    case "name":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            A short label to identify this simulated user profile.
          </Text>
          <Box>
            <Text>Label: </Text>
            <TextInput
              value={nameInput}
              onChange={setNameInput}
              onSubmit={(v) => {
                if (v.trim()) {
                  setNameVal(v.trim());
                  setStep("characteristics");
                }
              }}
              placeholder="e.g. Shy pregnant mother"
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>Press esc to cancel</Text>
          </Box>
        </Box>
      );

    case "characteristics":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text>Describe who this person is and how they communicate:</Text>
          <Text dimColor>
            This shapes how the simulated user talks, responds, and behaves
            during a simulation.
          </Text>
          <Box marginTop={1}>
            <TextArea
              value={characteristics}
              onChange={setChar}
              onSubmit={() => {
                if (characteristics.trim()) setStep("gender");
              }}
              placeholder="Name, age, occupation, personality, speaking style..."
              height={6}
            />
          </Box>
        </Box>
      );

    case "gender":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            Used to select the appropriate TTS voice in voice simulations.
          </Text>
          <Text>Gender:</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "Male", value: "male" },
                { label: "Female", value: "female" },
              ]}
              onSelect={(v) => {
                setGender(v);
                setStep("language");
              }}
            />
          </Box>
        </Box>
      );

    case "language":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            The language the simulated user will speak during the conversation.
          </Text>
          <Text>Language:</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "English", value: "english" },
                { label: "Hindi", value: "hindi" },
                { label: "Kannada", value: "kannada" },
              ]}
              onSelect={(v) => {
                setLang(v);
                setStep("interruption");
              }}
            />
          </Box>
        </Box>
      );

    case "interruption":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            How often the simulated user will interrupt the agent mid-sentence
            (voice simulations only).
          </Text>
          <Text>Interruption sensitivity:</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "None (0%)", value: "none" },
                { label: "Low (25%)", value: "low" },
                { label: "Medium (50%)", value: "medium" },
                { label: "High (80%)", value: "high" },
              ]}
              onSelect={(v) => {
                const persona = addPersona({
                  name,
                  characteristics: characteristics.trim(),
                  gender,
                  language,
                  interruption_sensitivity: v,
                });
                onComplete(persona);
              }}
            />
          </Box>
        </Box>
      );

    default:
      return null;
  }
}

// ═══════════════════════════════════════════════════════════════
// Scenario Creation
// ═══════════════════════════════════════════════════════════════
export function CreateScenarioFlow({
  onComplete,
  onCancel,
}: {
  onComplete: (scenario: StoredScenario) => void;
  onCancel: () => void;
}) {
  const [step, setStep] = useState<"name" | "description">("name");
  const [name, setNameVal] = useState("");
  const [desc, setDesc] = useState("");
  const [nameInput, setNameInput] = useState("");

  const header = (
    <Box marginBottom={1}>
      <Text bold color="cyan">
        Create Scenario
      </Text>
    </Box>
  );

  if (step === "name") {
    return (
      <Box flexDirection="column" padding={1}>
        {header}
        <Text dimColor>A short label for this conversation scenario.</Text>
        <Box>
          <Text>Label: </Text>
          <TextInput
            value={nameInput}
            onChange={setNameInput}
            onSubmit={(v) => {
              if (v.trim()) {
                setNameVal(v.trim());
                setStep("description");
              }
            }}
            placeholder="e.g. Complete health screening"
          />
        </Box>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" padding={1}>
      {header}
      <Text>Describe what the user should accomplish:</Text>
      <Text dimColor>
        The simulated user will follow this goal during the conversation to test
        how your agent handles it.
      </Text>
      <Box marginTop={1}>
        <TextArea
          value={desc}
          onChange={setDesc}
          onSubmit={() => {
            if (desc.trim()) {
              const scenario = addScenario({ name, description: desc.trim() });
              onComplete(scenario);
            }
          }}
          placeholder="The user should..."
          height={4}
        />
      </Box>
    </Box>
  );
}

// ═══════════════════════════════════════════════════════════════
// Metric Creation
// ═══════════════════════════════════════════════════════════════
export function CreateMetricFlow({
  onComplete,
  onCancel,
}: {
  onComplete: (metric: StoredMetric) => void;
  onCancel: () => void;
}) {
  const [step, setStep] = useState<"name" | "description">("name");
  const [name, setNameVal] = useState("");
  const [desc, setDesc] = useState("");
  const [nameInput, setNameInput] = useState("");

  const header = (
    <Box marginBottom={1}>
      <Text bold color="cyan">
        Create Metric
      </Text>
    </Box>
  );

  if (step === "name") {
    return (
      <Box flexDirection="column" padding={1}>
        {header}
        <Text dimColor>
          A name for this evaluation criterion (e.g. "completeness", "empathy").
        </Text>
        <Box>
          <Text>Metric name: </Text>
          <TextInput
            value={nameInput}
            onChange={setNameInput}
            onSubmit={(v) => {
              if (v.trim()) {
                setNameVal(v.trim());
                setStep("description");
              }
            }}
            placeholder="e.g. completeness"
          />
        </Box>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" padding={1}>
      {header}
      <Text>Evaluation instructions for the LLM judge:</Text>
      <Text dimColor>
        Describe what "pass" vs "fail" looks like — this is sent to an LLM judge
        to score each simulation.
      </Text>
      <Box marginTop={1}>
        <TextArea
          value={desc}
          onChange={setDesc}
          onSubmit={() => {
            if (desc.trim()) {
              const metric = addMetric({ name, description: desc.trim() });
              onComplete(metric);
            }
          }}
          placeholder="Define pass/fail criteria..."
          height={4}
        />
      </Box>
    </Box>
  );
}

// ═══════════════════════════════════════════════════════════════
// Agent Creation
// ═══════════════════════════════════════════════════════════════
type AgentStep =
  | "name"
  | "system-prompt"
  | "select-tools"
  | "create-tool"
  | "speaks-first"
  | "max-turns";

export function CreateAgentFlow({
  onComplete,
  onCancel,
}: {
  onComplete: (agent: StoredAgent) => void;
  onCancel: () => void;
}) {
  const [step, setStep] = useState<AgentStep>("name");
  const [name, setNameVal] = useState("");
  const [prompt, setPrompt] = useState("");
  const [tools, setTools] = useState<StoredTool[]>([]);
  const [speaksFirst, setSpeaksFirst] = useState(true);
  const [maxTurns, setMaxTurns] = useState("50");
  const [nameInput, setNameInput] = useState("");
  const [turnsInput, setTurnsInput] = useState("50");

  const header = (
    <Box marginBottom={1}>
      <Text bold color="cyan">
        Create Agent
      </Text>
    </Box>
  );

  switch (step) {
    case "name":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            A friendly name to identify this agent configuration.
          </Text>
          <Box>
            <Text>Agent name: </Text>
            <TextInput
              value={nameInput}
              onChange={setNameInput}
              onSubmit={(v) => {
                if (v.trim()) {
                  setNameVal(v.trim());
                  setStep("system-prompt");
                }
              }}
              placeholder="e.g. Health Screening Bot"
            />
          </Box>
        </Box>
      );

    case "system-prompt":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text>System prompt (instructions for the agent):</Text>
          <Text dimColor>
            This is the core instruction set that tells the LLM how to behave as
            your agent.
          </Text>
          <Box marginTop={1}>
            <TextArea
              value={prompt}
              onChange={setPrompt}
              onSubmit={() => {
                if (prompt.trim()) setStep("select-tools");
              }}
              placeholder="You are a helpful assistant that..."
              height={8}
            />
          </Box>
        </Box>
      );

    case "select-tools":
      return (
        <ToolPickStep
          onComplete={(selected) => {
            setTools(selected);
            setStep("speaks-first");
          }}
          onSkip={() => setStep("speaks-first")}
        />
      );

    case "create-tool":
      return (
        <CreateToolFlow
          onComplete={(tool) => {
            setTools((prev) => [...prev, tool]);
            setStep("select-tools");
          }}
          onCancel={() => setStep("select-tools")}
        />
      );

    case "speaks-first":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            Choose whether the agent initiates the conversation or waits for the
            user to speak.
          </Text>
          <Text>Who speaks first?</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "Agent speaks first", value: "yes" },
                { label: "User speaks first", value: "no" },
              ]}
              onSelect={(v) => {
                setSpeaksFirst(v === "yes");
                setStep("max-turns");
              }}
            />
          </Box>
        </Box>
      );

    case "max-turns":
      return (
        <Box flexDirection="column" padding={1}>
          {header}
          <Text dimColor>
            After this many assistant turns, the simulation will end
            automatically.
          </Text>
          <Box>
            <Text>Max assistant turns: </Text>
            <TextInput
              value={turnsInput}
              onChange={setTurnsInput}
              onSubmit={(v) => {
                const n = parseInt(v) || 50;
                const agent = addAgent({
                  name,
                  system_prompt: prompt.trim(),
                  tools,
                  settings: { agent_speaks_first: speaksFirst, max_turns: n },
                });
                onComplete(agent);
              }}
              placeholder="50"
            />
          </Box>
          <Box marginTop={1}>
            <Text dimColor>Press enter to confirm (default: 50)</Text>
          </Box>
        </Box>
      );

    default:
      return null;
  }
}

// ═══════════════════════════════════════════════════════════════
// Tool Selection (multi-select existing + create new)
// ═══════════════════════════════════════════════════════════════
function ToolPickStep({
  onComplete,
  onSkip,
}: {
  onComplete: (tools: StoredTool[]) => void;
  onSkip: () => void;
}) {
  const [mode, setMode] = useState<"menu" | "select" | "create">("menu");
  const [selected, setSelected] = useState<StoredTool[]>([]);
  const existing = useMemo(() => listTools(), [mode]);

  if (mode === "create") {
    return (
      <CreateToolFlow
        onComplete={(tool) => {
          setSelected((prev) => [...prev, tool]);
          setMode("menu");
        }}
        onCancel={() => setMode("menu")}
      />
    );
  }

  if (mode === "select" && existing.length > 0) {
    return (
      <Box flexDirection="column" padding={1}>
        <Box marginBottom={1}>
          <Text bold color="cyan">
            Select Tools
          </Text>
          {selected.length > 0 && (
            <Text dimColor> ({selected.length} already added)</Text>
          )}
        </Box>
        <MultiSelect
          items={existing.map((t) => ({
            label: `${t.name} (${t.type})`,
            value: t.id,
          }))}
          onSubmit={(ids) => {
            const picked = existing.filter((t) => ids.includes(t.id));
            const merged = [...selected];
            for (const p of picked) {
              if (!merged.find((m) => m.id === p.id)) merged.push(p);
            }
            setSelected(merged);
            setMode("menu");
          }}
        />
      </Box>
    );
  }

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Tools
        </Text>
      </Box>
      {selected.length > 0 && (
        <Box flexDirection="column" marginBottom={1}>
          {selected.map((t) => (
            <Text key={t.id}>
              {" "}
              + {t.name} <Text dimColor>({t.type})</Text>
            </Text>
          ))}
        </Box>
      )}
      <SelectInput
        items={[
          ...(existing.length > 0
            ? [{ label: "Select from existing tools", value: "select" }]
            : []),
          { label: "Create new tool", value: "create" },
          {
            label:
              selected.length > 0
                ? `Done (${selected.length} tools)`
                : "Skip (no tools)",
            value: "done",
          },
        ]}
        onSelect={(v) => {
          if (v === "select") setMode("select");
          else if (v === "create") setMode("create");
          else if (selected.length > 0) onComplete(selected);
          else onSkip();
        }}
      />
    </Box>
  );
}

// ═══════════════════════════════════════════════════════════════
// Agent Selection (single select + create new)
// ═══════════════════════════════════════════════════════════════
export function AgentSelectStep({
  onComplete,
  onBack,
}: {
  onComplete: (agent: StoredAgent) => void;
  onBack?: () => void;
}) {
  const [creating, setCreating] = useState(false);
  const agents = useMemo(() => listAgents(), [creating]);

  if (creating) {
    return (
      <CreateAgentFlow
        onComplete={(agent) => onComplete(agent)}
        onCancel={() => setCreating(false)}
      />
    );
  }

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Select Agent
        </Text>
      </Box>
      <SelectInput
        items={[
          ...agents.map((a) => ({
            label: `${a.name} — ${a.tools.length} tools`,
            value: a.id,
          })),
          { label: "+ Create new agent", value: "_create" },
          ...(onBack ? [{ label: "Back", value: "_back" }] : []),
        ]}
        onSelect={(v) => {
          if (v === "_create") {
            setCreating(true);
            return;
          }
          if (v === "_back" && onBack) {
            onBack();
            return;
          }
          const agent = agents.find((a) => a.id === v);
          if (agent) onComplete(agent);
        }}
      />
    </Box>
  );
}

// ═══════════════════════════════════════════════════════════════
// Generic Resource Multi-Select (personas, scenarios, metrics)
// ═══════════════════════════════════════════════════════════════
export function ResourcePickStep({
  resourceType,
  title,
  onComplete,
  onBack,
}: {
  resourceType: "personas" | "scenarios" | "metrics";
  title: string;
  onComplete: (items: any[]) => void;
  onBack?: () => void;
}) {
  const [mode, setMode] = useState<"menu" | "select" | "create">("menu");
  const [selected, setSelected] = useState<any[]>([]);

  const loaders = {
    personas: listPersonas,
    scenarios: listScenarios,
    metrics: listMetrics,
  };
  const existing = useMemo(() => loaders[resourceType](), [mode]);

  if (mode === "create") {
    const CreationFlow = {
      personas: CreatePersonaFlow,
      scenarios: CreateScenarioFlow,
      metrics: CreateMetricFlow,
    }[resourceType];

    return (
      <CreationFlow
        onComplete={(item: any) => {
          setSelected((prev) => [...prev, item]);
          setMode("menu");
        }}
        onCancel={() => setMode("menu")}
      />
    );
  }

  if (mode === "select" && existing.length > 0) {
    return (
      <Box flexDirection="column" padding={1}>
        <Box marginBottom={1}>
          <Text bold color="cyan">
            Select {title}
          </Text>
        </Box>
        <MultiSelect
          items={existing.map((item: any) => ({
            label: item.name,
            value: item.id,
          }))}
          onSubmit={(ids) => {
            const picked = existing.filter((item: any) =>
              ids.includes(item.id)
            );
            const merged = [...selected];
            for (const p of picked) {
              if (!merged.find((m: any) => m.id === p.id)) merged.push(p);
            }
            setSelected(merged);
            setMode("menu");
          }}
        />
      </Box>
    );
  }

  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          {title}
        </Text>
      </Box>
      {selected.length > 0 && (
        <Box flexDirection="column" marginBottom={1}>
          {selected.map((item: any) => (
            <Text key={item.id}> + {item.name}</Text>
          ))}
        </Box>
      )}
      <SelectInput
        items={[
          ...(existing.length > 0
            ? [
                {
                  label: `Select from existing ${resourceType}`,
                  value: "select",
                },
              ]
            : []),
          { label: `Create new`, value: "create" },
          ...(selected.length > 0
            ? [
                {
                  label: `Continue (${selected.length} selected)`,
                  value: "done",
                },
              ]
            : []),
          ...(onBack ? [{ label: "Back", value: "_back" }] : []),
        ]}
        onSelect={(v) => {
          if (v === "select") setMode("select");
          else if (v === "create") setMode("create");
          else if (v === "_back" && onBack) onBack();
          else if (v === "done") onComplete(selected);
        }}
      />
      {selected.length === 0 && existing.length === 0 && (
        <Box marginTop={1}>
          <Text dimColor>No {resourceType} found. Create one to continue.</Text>
        </Box>
      )}
    </Box>
  );
}

// ═══════════════════════════════════════════════════════════════
// Resource Management Screens (for main menu)
// ═══════════════════════════════════════════════════════════════
export function ResourceListScreen({
  resourceType,
  title,
  onBack,
}: {
  resourceType: "agents" | "tools" | "personas" | "scenarios" | "metrics";
  title: string;
  onBack: () => void;
}) {
  const [mode, setMode] = useState<"list" | "create" | "detail" | "edit">(
    "list"
  );
  const [refreshKey, setRefreshKey] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [editField, setEditField] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");

  const loaders: Record<string, () => any[]> = {
    agents: listAgents,
    tools: listTools,
    personas: listPersonas,
    scenarios: listScenarios,
    metrics: listMetrics,
  };
  const removers: Record<string, (id: string) => void> = {
    agents: removeAgent,
    tools: removeTool,
    personas: removePersona,
    scenarios: removeScenario,
    metrics: removeMetric,
  };
  const updaters: Record<string, (id: string, updates: any) => void> = {
    agents: updateAgent,
    tools: updateTool,
    personas: updatePersona,
    scenarios: updateScenario,
    metrics: updateMetric,
  };

  const items = useMemo(
    () => loaders[resourceType](),
    [refreshKey, mode, resourceType]
  );
  const selectedItem = selectedId
    ? items.find((i: any) => i.id === selectedId)
    : null;

  // ── Create mode ──
  if (mode === "create") {
    const flows: Record<string, React.FC<any>> = {
      agents: CreateAgentFlow,
      tools: CreateToolFlow,
      personas: CreatePersonaFlow,
      scenarios: CreateScenarioFlow,
      metrics: CreateMetricFlow,
    };
    const Flow = flows[resourceType]!;
    return (
      <Flow
        onComplete={() => {
          setMode("list");
          setRefreshKey((k) => k + 1);
        }}
        onCancel={() => setMode("list")}
      />
    );
  }

  // ── Edit field mode ──
  if (mode === "edit" && selectedItem && editField) {
    const isTextArea = [
      "system_prompt",
      "characteristics",
      "description",
    ].includes(editField);

    if (editField === "agent_speaks_first") {
      return (
        <Box flexDirection="column" padding={1}>
          <Box marginBottom={1}>
            <Text bold color="cyan">
              Edit — {selectedItem.name}
            </Text>
          </Box>
          <Text>Who speaks first?</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "Agent speaks first", value: "yes" },
                { label: "User speaks first", value: "no" },
              ]}
              onSelect={(v) => {
                updaters[resourceType](selectedItem.id, {
                  settings: {
                    ...selectedItem.settings,
                    agent_speaks_first: v === "yes",
                  },
                });
                setRefreshKey((k) => k + 1);
                setMode("detail");
              }}
            />
          </Box>
        </Box>
      );
    }

    if (editField === "max_turns") {
      return (
        <Box flexDirection="column" padding={1}>
          <Box marginBottom={1}>
            <Text bold color="cyan">
              Edit — {selectedItem.name}
            </Text>
          </Box>
          <Box>
            <Text>Max assistant turns: </Text>
            <TextInput
              value={editValue}
              onChange={setEditValue}
              onSubmit={(v) => {
                const n = parseInt(v) || selectedItem.settings?.max_turns || 50;
                updaters[resourceType](selectedItem.id, {
                  settings: { ...selectedItem.settings, max_turns: n },
                });
                setRefreshKey((k) => k + 1);
                setMode("detail");
              }}
            />
          </Box>
        </Box>
      );
    }

    if (editField === "gender") {
      return (
        <Box flexDirection="column" padding={1}>
          <Box marginBottom={1}>
            <Text bold color="cyan">
              Edit — {selectedItem.name}
            </Text>
          </Box>
          <Text>Gender:</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "Male", value: "male" },
                { label: "Female", value: "female" },
              ]}
              onSelect={(v) => {
                updaters[resourceType](selectedItem.id, { gender: v });
                setRefreshKey((k) => k + 1);
                setMode("detail");
              }}
            />
          </Box>
        </Box>
      );
    }

    if (editField === "language") {
      return (
        <Box flexDirection="column" padding={1}>
          <Box marginBottom={1}>
            <Text bold color="cyan">
              Edit — {selectedItem.name}
            </Text>
          </Box>
          <Text>Language:</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "English", value: "english" },
                { label: "Hindi", value: "hindi" },
                { label: "Kannada", value: "kannada" },
              ]}
              onSelect={(v) => {
                updaters[resourceType](selectedItem.id, { language: v });
                setRefreshKey((k) => k + 1);
                setMode("detail");
              }}
            />
          </Box>
        </Box>
      );
    }

    if (editField === "interruption_sensitivity") {
      return (
        <Box flexDirection="column" padding={1}>
          <Box marginBottom={1}>
            <Text bold color="cyan">
              Edit — {selectedItem.name}
            </Text>
          </Box>
          <Text>Interruption sensitivity:</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "None (0%)", value: "none" },
                { label: "Low (25%)", value: "low" },
                { label: "Medium (50%)", value: "medium" },
                { label: "High (80%)", value: "high" },
              ]}
              onSelect={(v) => {
                updaters[resourceType](selectedItem.id, {
                  interruption_sensitivity: v,
                });
                setRefreshKey((k) => k + 1);
                setMode("detail");
              }}
            />
          </Box>
        </Box>
      );
    }

    if (editField === "type") {
      return (
        <Box flexDirection="column" padding={1}>
          <Box marginBottom={1}>
            <Text bold color="cyan">
              Edit — {selectedItem.name}
            </Text>
          </Box>
          <Text>Tool type:</Text>
          <Box marginTop={1}>
            <SelectInput
              items={[
                { label: "Structured Output", value: "structured_output" },
                { label: "Webhook", value: "webhook" },
              ]}
              onSelect={(v) => {
                updaters[resourceType](selectedItem.id, { type: v });
                setRefreshKey((k) => k + 1);
                setMode("detail");
              }}
            />
          </Box>
        </Box>
      );
    }

    // TextArea or TextInput for text fields
    if (isTextArea) {
      return (
        <Box flexDirection="column" padding={1}>
          <Box marginBottom={1}>
            <Text bold color="cyan">
              Edit — {selectedItem.name}
            </Text>
            <Text dimColor> — {editField}</Text>
          </Box>
          <TextArea
            value={editValue}
            onChange={setEditValue}
            onSubmit={() => {
              if (editValue.trim()) {
                updaters[resourceType](selectedItem.id, {
                  [editField]: editValue.trim(),
                });
                setRefreshKey((k) => k + 1);
              }
              setMode("detail");
            }}
            height={8}
          />
        </Box>
      );
    }

    // Default: TextInput for simple text fields
    return (
      <Box flexDirection="column" padding={1}>
        <Box marginBottom={1}>
          <Text bold color="cyan">
            Edit — {selectedItem.name}
          </Text>
          <Text dimColor> — {editField}</Text>
        </Box>
        <Box>
          <TextInput
            value={editValue}
            onChange={setEditValue}
            onSubmit={(v) => {
              if (v.trim()) {
                updaters[resourceType](selectedItem.id, {
                  [editField]: v.trim(),
                });
                setRefreshKey((k) => k + 1);
              }
              setMode("detail");
            }}
          />
        </Box>
        <Box marginTop={1}>
          <Text dimColor>Press enter to save</Text>
        </Box>
      </Box>
    );
  }

  // ── Detail view ──
  if (mode === "detail" && selectedItem) {
    const fieldDefs = getFieldDefs(resourceType, selectedItem);

    return (
      <Box flexDirection="column" padding={1}>
        <Box marginBottom={1}>
          <Text bold color="cyan">
            {selectedItem.name}
          </Text>
        </Box>

        {fieldDefs.map((f) => (
          <Box key={f.key} marginBottom={0}>
            <Text bold>{f.label}: </Text>
            <Text>
              {f.display.length > 80
                ? f.display.slice(0, 80) + "..."
                : f.display}
            </Text>
          </Box>
        ))}

        <Box marginTop={1}>
          <SelectInput
            items={[
              ...fieldDefs
                .filter((f) => f.editable)
                .map((f) => ({
                  label: `Edit ${f.label.toLowerCase()}`,
                  value: `edit:${f.key}`,
                })),
              { label: "Delete", value: "delete" },
              { label: "Back to list", value: "back" },
            ]}
            onSelect={(v) => {
              if (v.startsWith("edit:")) {
                const key = v.slice(5);
                const current = getFieldValue(selectedItem, key);
                setEditField(key);
                setEditValue(current);
                setMode("edit");
              } else if (v === "delete") {
                removers[resourceType](selectedItem.id);
                setSelectedId(null);
                setRefreshKey((k) => k + 1);
                setMode("list");
              } else {
                setMode("list");
              }
            }}
          />
        </Box>
      </Box>
    );
  }

  // ── List view ──
  return (
    <Box flexDirection="column" padding={1}>
      <Box marginBottom={1}>
        <Text bold color="cyan">
          {title}
        </Text>
        <Text dimColor> ({items.length} items)</Text>
      </Box>
      {items.length === 0 && (
        <Box marginBottom={1}>
          <Text dimColor>No {resourceType} found.</Text>
        </Box>
      )}
      <SelectInput
        items={[
          ...items.map((item: any) => ({
            label: formatListItem(resourceType, item),
            value: item.id,
          })),
          { label: "+ Create new", value: "_create" },
          { label: "Back to menu", value: "_back" },
        ]}
        onSelect={(v) => {
          if (v === "_create") setMode("create");
          else if (v === "_back") onBack();
          else {
            setSelectedId(v);
            setMode("detail");
          }
        }}
      />
    </Box>
  );
}

// ─── Helpers for detail view ─────────────────────────────────

interface FieldInfo {
  key: string;
  label: string;
  display: string;
  editable: boolean;
}

function getFieldValue(item: any, key: string): string {
  if (key === "agent_speaks_first")
    return item.settings?.agent_speaks_first ? "yes" : "no";
  if (key === "max_turns") return String(item.settings?.max_turns ?? 50);
  return String(item[key] ?? "");
}

function getFieldDefs(resourceType: string, item: any): FieldInfo[] {
  switch (resourceType) {
    case "agents":
      return [
        { key: "name", label: "Name", display: item.name, editable: true },
        {
          key: "system_prompt",
          label: "System prompt",
          display: item.system_prompt || "(empty)",
          editable: true,
        },
        {
          key: "tools",
          label: "Tools",
          display:
            item.tools?.length > 0
              ? item.tools.map((t: any) => t.name).join(", ")
              : "(none)",
          editable: false,
        },
        {
          key: "agent_speaks_first",
          label: "Who speaks first",
          display: item.settings?.agent_speaks_first ? "Agent" : "User",
          editable: true,
        },
        {
          key: "max_turns",
          label: "Max assistant turns",
          display: String(item.settings?.max_turns ?? 50),
          editable: true,
        },
      ];
    case "tools":
      return [
        { key: "name", label: "Name", display: item.name, editable: true },
        { key: "type", label: "Type", display: item.type, editable: true },
        {
          key: "description",
          label: "Description",
          display: item.description || "(empty)",
          editable: true,
        },
        {
          key: "parameters",
          label: "Parameters",
          display:
            item.parameters?.length > 0
              ? item.parameters.map((p: any) => p.id).join(", ")
              : "(none)",
          editable: false,
        },
        ...(item.type === "webhook" && item.webhook
          ? [
              {
                key: "webhook_url",
                label: "Webhook URL",
                display: item.webhook?.url || "(not set)",
                editable: false,
              },
              {
                key: "webhook_method",
                label: "Method",
                display: item.webhook?.method || "POST",
                editable: false,
              },
            ]
          : []),
      ];
    case "personas":
      return [
        { key: "name", label: "Name", display: item.name, editable: true },
        {
          key: "characteristics",
          label: "Characteristics",
          display: item.characteristics || "(empty)",
          editable: true,
        },
        {
          key: "gender",
          label: "Gender",
          display: item.gender,
          editable: true,
        },
        {
          key: "language",
          label: "Language",
          display: item.language,
          editable: true,
        },
        {
          key: "interruption_sensitivity",
          label: "Interruption",
          display: item.interruption_sensitivity,
          editable: true,
        },
      ];
    case "scenarios":
      return [
        { key: "name", label: "Name", display: item.name, editable: true },
        {
          key: "description",
          label: "Description",
          display: item.description || "(empty)",
          editable: true,
        },
      ];
    case "metrics":
      return [
        { key: "name", label: "Name", display: item.name, editable: true },
        {
          key: "description",
          label: "Description",
          display: item.description || "(empty)",
          editable: true,
        },
      ];
    default:
      return [];
  }
}

function formatListItem(resourceType: string, item: any): string {
  switch (resourceType) {
    case "agents":
      return `${item.name} — ${item.tools?.length ?? 0} tools`;
    case "tools":
      return `${item.name} (${item.type})`;
    case "personas":
      return `${item.name} — ${item.language}, ${item.gender}`;
    case "scenarios":
      return `${item.name}`;
    case "metrics":
      return `${item.name}`;
    default:
      return item.name;
  }
}
