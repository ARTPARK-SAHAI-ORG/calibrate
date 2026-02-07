import React, { useState, useEffect } from "react";
import { Box, Text, useInput } from "ink";

// ═══════════════════════════════════════════════════════════════
// Spinner
// ═══════════════════════════════════════════════════════════════
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export function Spinner({ label }: { label?: string }) {
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => {
      setFrame((prev) => (prev + 1) % SPINNER_FRAMES.length);
    }, 80);
    return () => clearInterval(timer);
  }, []);

  return (
    <Text>
      <Text color="cyan">{SPINNER_FRAMES[frame]}</Text>
      {label ? ` ${label}` : ""}
    </Text>
  );
}

// ═══════════════════════════════════════════════════════════════
// TextInput
// ═══════════════════════════════════════════════════════════════
interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit?: (value: string) => void;
  placeholder?: string;
  mask?: string;
  isActive?: boolean;
}

export function TextInput({
  value,
  onChange,
  onSubmit,
  placeholder,
  mask,
  isActive = true,
}: TextInputProps) {
  useInput(
    (input, key) => {
      if (key.return) {
        onSubmit?.(value);
        return;
      }
      if (key.backspace || key.delete) {
        onChange(value.slice(0, -1));
        return;
      }
      if (key.tab) return;
      if (input && !key.ctrl && !key.meta) {
        onChange(value + input);
      }
    },
    { isActive }
  );

  const displayValue = value ? (mask ? mask.repeat(value.length) : value) : "";

  return (
    <Box flexDirection="column">
      <Box>
        {displayValue ? (
          <Text>{displayValue}</Text>
        ) : placeholder ? (
          <Text dimColor>{placeholder}</Text>
        ) : null}
        {isActive && <Text color="cyan">{"▎"}</Text>}
      </Box>
    </Box>
  );
}

// ═══════════════════════════════════════════════════════════════
// MultiSelect
// ═══════════════════════════════════════════════════════════════
interface MultiSelectItem {
  label: string;
  value: string;
}

export function MultiSelect({
  items,
  onSubmit,
}: {
  items: MultiSelectItem[];
  onSubmit: (selected: string[]) => void;
}) {
  const [cursor, setCursor] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  useInput((input, key) => {
    if (key.upArrow) {
      setCursor((prev) => (prev > 0 ? prev - 1 : items.length - 1));
    } else if (key.downArrow) {
      setCursor((prev) => (prev < items.length - 1 ? prev + 1 : 0));
    } else if (input === " ") {
      setSelected((prev) => {
        const next = new Set(prev);
        const val = items[cursor]!.value;
        if (next.has(val)) next.delete(val);
        else next.add(val);
        return next;
      });
    } else if (key.return) {
      if (selected.size > 0) {
        onSubmit(Array.from(selected));
      }
    } else if (input === "a") {
      setSelected((prev) => {
        if (prev.size === items.length) return new Set();
        return new Set(items.map((i) => i.value));
      });
    }
  });

  return (
    <Box flexDirection="column">
      {items.map((item, i) => {
        const isSelected = selected.has(item.value);
        const isCursor = i === cursor;
        return (
          <Box key={item.value}>
            <Text color={isCursor ? "cyan" : undefined} bold={isCursor}>
              {isCursor ? " > " : "   "}
            </Text>
            <Text color={isSelected ? "green" : "gray"}>
              {isSelected ? "[x]" : "[ ]"}
            </Text>
            <Text color={isCursor ? "cyan" : undefined} bold={isCursor}>
              {" "}
              {item.label}
            </Text>
          </Box>
        );
      })}
      <Box marginTop={1}>
        <Text dimColor>
          {"  Space for toggle  a for all  Enter to confirm"}
        </Text>
        {selected.size > 0 && (
          <Text color="cyan"> ({selected.size} selected)</Text>
        )}
      </Box>
    </Box>
  );
}

// ═══════════════════════════════════════════════════════════════
// SelectInput (single select)
// ═══════════════════════════════════════════════════════════════
export function SelectInput({
  items,
  onSelect,
  initialIndex = 0,
}: {
  items: { label: string; value: string }[];
  onSelect: (value: string) => void;
  initialIndex?: number;
}) {
  const [cursor, setCursor] = useState(initialIndex);

  useInput((_input, key) => {
    if (key.upArrow) {
      setCursor((prev) => (prev > 0 ? prev - 1 : items.length - 1));
    } else if (key.downArrow) {
      setCursor((prev) => (prev < items.length - 1 ? prev + 1 : 0));
    } else if (key.return) {
      onSelect(items[cursor]!.value);
    }
  });

  return (
    <Box flexDirection="column">
      {items.map((item, i) => (
        <Box key={item.value}>
          <Text color={i === cursor ? "cyan" : undefined} bold={i === cursor}>
            {i === cursor ? " > " : "   "}
            {item.label}
          </Text>
        </Box>
      ))}
      <Box marginTop={1}>
        <Text dimColor>{"  ↑↓ navigate  Enter select"}</Text>
      </Box>
    </Box>
  );
}

// ═══════════════════════════════════════════════════════════════
// TextArea (multi-line text input)
// ═══════════════════════════════════════════════════════════════
interface TextAreaProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit?: (value: string) => void;
  placeholder?: string;
  isActive?: boolean;
  height?: number;
}

export function TextArea({
  value,
  onChange,
  onSubmit,
  placeholder,
  isActive = true,
  height = 6,
}: TextAreaProps) {
  useInput(
    (input, key) => {
      if (key.escape) {
        onSubmit?.(value);
        return;
      }
      if (key.return) {
        onChange(value + "\n");
        return;
      }
      if (key.backspace || key.delete) {
        onChange(value.slice(0, -1));
        return;
      }
      if (key.tab) return;
      if (input && !key.ctrl && !key.meta) {
        onChange(value + input);
      }
    },
    { isActive }
  );

  const lines = value ? value.split("\n") : [];
  const displayLines = lines.length > 0 ? lines.slice(-height) : [];

  return (
    <Box flexDirection="column">
      <Box
        flexDirection="column"
        borderStyle="single"
        borderColor={isActive ? "cyan" : "gray"}
        paddingX={1}
        width={60}
        height={height + 2}
      >
        {displayLines.length > 0 ? (
          displayLines.map((line, i) => (
            <Text key={i}>
              {line}
              {i === displayLines.length - 1 && isActive ? (
                <Text color="cyan">{"▎"}</Text>
              ) : null}
            </Text>
          ))
        ) : placeholder ? (
          <Text dimColor>{placeholder}</Text>
        ) : null}
        {displayLines.length === 0 && isActive && !placeholder && (
          <Text color="cyan">{"▎"}</Text>
        )}
      </Box>
      {isActive && <Text dimColor>{"  enter: new line  esc: done"}</Text>}
    </Box>
  );
}

// ═══════════════════════════════════════════════════════════════
// Table
// ═══════════════════════════════════════════════════════════════
interface Column {
  key: string;
  label: string;
  width: number;
  align?: "left" | "right";
}

function pad(
  str: string,
  width: number,
  align: "left" | "right" = "left"
): string {
  const s = str.slice(0, width);
  return align === "right" ? s.padStart(width) : s.padEnd(width);
}

export function Table({
  columns,
  data,
}: {
  columns: Column[];
  data: Record<string, string>[];
}) {
  return (
    <Box flexDirection="column">
      {/* Header */}
      <Box>
        {columns.map((col, i) => (
          <Text key={col.key} bold>
            {i > 0 ? " | " : " "}
            {pad(col.label, col.width)}
          </Text>
        ))}
      </Box>
      {/* Separator */}
      <Text dimColor>
        {" " + columns.map((c) => "\u2500".repeat(c.width)).join("-+-")}
      </Text>
      {/* Rows */}
      {data.map((row, rowIdx) => (
        <Box key={rowIdx}>
          {columns.map((col, i) => (
            <Text key={col.key}>
              {i > 0 ? " | " : " "}
              {pad(row[col.key] || "", col.width, col.align)}
            </Text>
          ))}
        </Box>
      ))}
    </Box>
  );
}

// ═══════════════════════════════════════════════════════════════
// BarChart
// ═══════════════════════════════════════════════════════════════
export function BarChart({
  data,
  maxWidth = 25,
}: {
  data: { label: string; value: number; color?: string }[];
  maxWidth?: number;
}) {
  const maxValue = Math.max(...data.map((d) => d.value), 0.001);
  const maxLabelWidth = Math.max(...data.map((d) => d.label.length));

  return (
    <Box flexDirection="column">
      {data.map((d) => {
        const barWidth = Math.max(
          1,
          Math.round((d.value / maxValue) * maxWidth)
        );
        const emptyWidth = maxWidth - barWidth;
        return (
          <Box key={d.label} gap={1}>
            <Text>{d.label.padEnd(maxLabelWidth)}</Text>
            <Text color={d.color || "green"}>{"\u2588".repeat(barWidth)}</Text>
            <Text dimColor>{"\u2591".repeat(emptyWidth)}</Text>
            <Text bold> {d.value.toFixed(2)}</Text>
          </Box>
        );
      })}
    </Box>
  );
}
