const fs = require('fs');

const CONTEXT_FILE = '.cursor/rules/app-details.md';
const README_FILE = 'README.md';
const DOCS_PATH = 'docs';

async function main() {
  const input = JSON.parse(await readStdin());

  if (input.status !== 'completed' || input.loop_count >= 1) {
    console.log(JSON.stringify({}));
    return;
  }

  // Only trigger if codebase-context.md was NOT the only file changed
  console.log(JSON.stringify({
    followup_message: `Only go ahead if you made code changes (not just updated ${CONTEXT_FILE}):
    
    Update ${CONTEXT_FILE} with any new information about the codebase from the changes you just made. Include:
- What changed
- Why it changed
- New patterns or conventions introduced
- Dependencies or relationships affected
- Any gotchas or edge cases discovered

Don't maintain it as a changelog. Instead, look for what was already present but has become obsolete now. Remove or update it as necessary. Look for what is missing and add it.

Update ${README_FILE} if any changes were made that conflict with what is present in the ${README_FILE} relevant to the changes made or if any additions were made to the core functionality of the codebase that are not present in the ${README_FILE}.

Update ${DOCS_PATH} if any changes were made that conflict with what is present in the ${DOCS_PATH} relevant to the changes made or if any additions were made to the core functionality of the codebase that are not present in the ${DOCS_PATH}.

Keep the files organized and don't remove existing relevant context.`
  }));
}

function readStdin() {
  return new Promise(resolve => {
    let data = '';
    process.stdin.on('data', chunk => data += chunk);
    process.stdin.on('end', () => resolve(data));
  });
}

main();