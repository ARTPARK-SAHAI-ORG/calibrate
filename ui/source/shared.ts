import { execSync } from 'node:child_process';
import net from 'node:net';
import fs from 'node:fs';
import path from 'node:path';

export interface CalibrateCmd {
  cmd: string;
  args: string[];
}

export function findCalibrateBin(): CalibrateCmd | null {
  try {
    execSync('which calibrate', { stdio: 'pipe' });
    return { cmd: 'calibrate', args: [] };
  } catch {}

  for (const rel of ['../.venv/bin/calibrate', '.venv/bin/calibrate']) {
    const abs = path.resolve(rel);
    if (fs.existsSync(abs)) {
      return { cmd: abs, args: [] };
    }
  }

  try {
    execSync('uv run which calibrate', { stdio: 'pipe', cwd: path.resolve('..') });
    return { cmd: 'uv', args: ['run', 'calibrate'] };
  } catch {}

  return null;
}

/**
 * Check if a port is available (not in use).
 * Returns a promise that resolves to true if the port is available.
 */
export function isPortAvailable(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', () => {
      resolve(false);
    });
    server.once('listening', () => {
      server.close();
      resolve(true);
    });
    server.listen(port, '127.0.0.1');
  });
}

/**
 * Find the next available port starting from the given port.
 * @param startPort - The port to start searching from.
 * @param maxAttempts - Maximum number of ports to check (default 100).
 * @returns The first available port, or null if none found.
 */
export async function findAvailablePort(
  startPort: number,
  maxAttempts: number = 100
): Promise<number | null> {
  for (let i = 0; i < maxAttempts; i++) {
    const port = startPort + i;
    const available = await isPortAvailable(port);
    if (available) {
      return port;
    }
  }
  return null;
}

export function stripAnsi(str: string): string {
  return str.replace(/\x1b\[[0-9;]*m/g, '');
}

export type AppMode =
  | 'menu'
  | 'stt'
  | 'tts'
  | 'llm'
  | 'simulations';
