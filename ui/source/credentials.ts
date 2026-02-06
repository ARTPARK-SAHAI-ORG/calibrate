import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const CRED_DIR = path.join(os.homedir(), '.calibrate');
const CRED_FILE = path.join(CRED_DIR, 'credentials.json');

export function loadCredentials(): Record<string, string> {
  try {
    return JSON.parse(fs.readFileSync(CRED_FILE, 'utf-8'));
  } catch {
    return {};
  }
}

export function saveCredential(key: string, value: string): void {
  const creds = loadCredentials();
  creds[key] = value;
  fs.mkdirSync(CRED_DIR, {recursive: true, mode: 0o700});
  fs.writeFileSync(CRED_FILE, JSON.stringify(creds, null, 2), {mode: 0o600});
}

export function getCredential(key: string): string | undefined {
  const creds = loadCredentials();
  return creds[key] || process.env[key] || undefined;
}
