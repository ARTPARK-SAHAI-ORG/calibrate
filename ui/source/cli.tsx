#!/usr/bin/env node
import React from "react";
import { render } from "ink";
import { App } from "./app.js";
import type { Mode } from "./app.js";

const mode = (process.argv[2] || "menu") as Mode;
render(<App mode={mode} />);
