# Running background-noise voice-agent experiments

End-to-end guide to reproduce the noise-injection experiments: stand up the agent
under test, set up Calibrate on this branch, run the easy/hard noise batches, and
read the results.

**Two repos are involved:**
| Repo | Role |
|---|---|
| **form-bharo** (`git@github.com:ARTPARK-SAHAI-ORG/form-bharo.git`) | the **agent under test** — the form-filling voice bot the simulated caller talks to |
| **calibrate** (this repo, branch `claude/festive-rosalind-92d0d6`) | the **evaluator** — drives a simulated noisy caller and scores the agent |

The flow: Calibrate spins up a simulated caller (with background noise mixed in),
connects to the form-bharo agent over a WebSocket, they hold a Hindi enrollment
conversation, and Calibrate saves audio + transcripts + scores. The agent, in
parallel, writes the **filled form** it captured to disk (`data.json`) — the real
ground truth.

---

## Part A — Set up the agent backend (form-bharo)

### 1. Clone and run the server
```bash
git clone git@github.com:ARTPARK-SAHAI-ORG/form-bharo.git
cd form-bharo/server/src
cp env.example .env          # then fill in the agent's provider keys (STT/TTS/LLM, e.g. SARVAM/GOOGLE/OPENAI)
uv run uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```
The env file **must** live at `form-bharo/server/src/.env`. Fill in the agent's
own provider keys (STT/TTS/LLM). **If you don't know what to put in it, ask Aman.**

The agent is now running on **http://localhost:7860**.

### 2. Create the `swasth-kadam-v5` agent
Create the agent folder and save the config below as its `config.json`:

```bash
mkdir -p server/db/agents/swasth-kadam-v5
# then create server/db/agents/swasth-kadam-v5/config.json with the JSON below
```

`server/db/agents/swasth-kadam-v5/config.json`:

```json
{
    "title": "Swasth kadam Health Survey V5",
    "questions": [
        {
            "name": "name",
            "label": "name",
            "type": "string",
            "required": false,
            "script": "\u0905\u092a\u0928\u093e \u092a\u0942\u0930\u093e \u0928\u093e\u092e \u092c\u0924\u093e\u0907\u090f",
            "validation": {
                "type": "min_length",
                "rules": {
                    "min_length": 2
                }
            },
            "retry_config": {
                "allow_retries": true,
                "until_answered": false,
                "max_retries": 1,
                "retry_messages": [
                    "\u0915\u0943\u092a\u092f\u093e \u0905\u092a\u0928\u093e \u092a\u0942\u0930\u093e \u0928\u093e\u092e \u092c\u0924\u093e\u0907\u090f\u0964"
                ],
                "exhausted_action": "skip"
            }
        },
        {
            "name": "district",
            "label": "district",
            "type": "single-select",
            "options": [
                "Mumbai",
                "Aurangabad",
                "Nashik",
                "Nagpur",
                "Pune"
            ],
            "required": true,
            "advanced_instructions": "IMPORTANT: On the initial ask for this question, say ONLY the question Script exactly. Do NOT add the district options on the initial ask, even though this is a single-select question.",
            "script": "\u0905\u092a\u0928\u0947 \u091c\u093f\u0932\u0947 \u0915\u093e \u0928\u093e\u092e \u092c\u0924\u093e\u090f\u0902",
            "retry_config": {
                "allow_retries": true,
                "until_answered": false,
                "max_retries": 2,
                "retry_messages": [
                    "\u0906\u092a\u0915\u093e \u0906\u0902\u0917\u0928\u0935\u093e\u0921\u093c\u0940 \u0915\u093f\u0938 \u091c\u093f\u0932\u0947 \u092e\u0947\u0902 \u0906\u0924\u093e \u0939\u0948? \u0915\u0943\u092a\u092f\u093e \u0907\u0928\u092e\u0947\u0902 \u0938\u0947 \u090f\u0915 \u091a\u0941\u0928\u0947\u0902: Mumbai, Aurangabad, Nashik, Nagpur ya Pune\u0964",
                    "\u0906\u092a\u0915\u093e \u0906\u0902\u0917\u0928\u0935\u093e\u0921\u093c\u0940 \u0915\u093f\u0938 \u091c\u093f\u0932\u0947 \u092e\u0947\u0902 \u0906\u0924\u093e \u0939\u0948? \u0915\u0943\u092a\u092f\u093e \u0907\u0928\u092e\u0947\u0902 \u0938\u0947 \u090f\u0915 \u091a\u0941\u0928\u0947\u0902: Mumbai, Aurangabad, Nashik, Nagpur ya Pune\u0964"
                ],
                "exhausted_action": "end_call"
            }
        },
        {
            "name": "anganwadi_name",
            "label": "anganwadi_name",
            "type": "string",
            "required": false,
            "skip_message": "\u0939\u092e \u0906\u0917\u0947 \u092c\u0922\u093c\u0924\u0947 \u0939\u0948\u0902\u0964",
            "script": "\u0905\u092a\u0928\u0947 \u0906\u0902\u0917\u0928\u0935\u093e\u0921\u093c\u0940 \u0915\u093e \u0928\u093e\u092e \u092c\u0924\u093e\u090f\u0902",
            "retry_config": {
                "allow_retries": true,
                "until_answered": false,
                "max_retries": 1,
                "retry_messages": [
                    "\u0905\u0917\u0930 \u0906\u092a\u0915\u094b \u0905\u092a\u0928\u0947 \u0906\u0902\u0917\u0928\u0935\u093e\u0921\u093c\u0940 \u0915\u093e \u0928\u093e\u092e \u0928\u0939\u0940\u0902 \u092a\u0924\u093e \u0939\u0948, \u0924\u094b \u0915\u0943\u092a\u092f\u093e \\\"\u0928\u0939\u0940\u0902 \u092a\u0924\u093e\\\" \u092c\u094b\u0932\u093f\u090f\u0964"
                ],
                "exhausted_action": "skip"
            }
        },
        {
            "name": "pregnant_or_not",
            "label": "pregnant_or_not",
            "type": "boolean",
            "required": true,
            "script": "\u0915\u094d\u092f\u093e \u0906\u092a \u0905\u092d\u0940 \u0917\u0930\u094d\u092d\u0935\u0924\u0940 \u0939\u0948\u0902?",
            "retry_config": {
                "allow_retries": true,
                "until_answered": false,
                "max_retries": 2,
                "retry_messages": [
                    "\u0915\u094d\u092f\u093e \u0906\u092a \u0905\u092d\u0940 \u0917\u0930\u094d\u092d\u0935\u0924\u0940 \u0939\u0948\u0902? \u0915\u0943\u092a\u092f\u093e \u0939\u093e\u0901 \u092f\u093e \u0928\u093e \u092e\u0947\u0902 \u091c\u0935\u093e\u092c \u0926\u0947\u0902\u0964",
                    "\u0915\u094d\u092f\u093e \u0906\u092a \u0905\u092d\u0940 \u0917\u0930\u094d\u092d\u0935\u0924\u0940 \u0939\u0948\u0902? \u0915\u0943\u092a\u092f\u093e \u0939\u093e\u0901 \u092f\u093e \u0928\u093e \u092e\u0947\u0902 \u091c\u0935\u093e\u092c \u0926\u0947\u0902\u0964"
                ],
                "exhausted_action": "end_call"
            }
        },
        {
            "name": "gestational_age_months",
            "label": "gestational_age_months",
            "type": "number",
            "number_format": "integer",
            "required": false,
            "advanced_instructions": "The answer is in number of months.",
            "script": "\u0906\u092a\u0915\u094b \u0917\u0930\u094d\u092d\u0935\u0924\u0940 \u0939\u0941\u090f \u0915\u093f\u0924\u0928\u0947 \u092e\u0939\u0940\u0928\u0947 \u0939\u0941\u090f \u0939\u0948\u0902?",
            "branch_id": "branch_pregnant",
            "validation": {
                "type": "number_range",
                "rules": {
                    "min": 1,
                    "max": 9
                }
            },
            "retry_config": {
                "allow_retries": true,
                "until_answered": false,
                "max_retries": 2,
                "retry_messages": [
                    "\u0906\u092a\u0915\u094b \u0917\u0930\u094d\u092d\u0935\u0924\u0940 \u0939\u0941\u090f \u0915\u093f\u0924\u0928\u0947 \u092e\u0939\u0940\u0928\u0947 \u0939\u0941\u090f \u0939\u0948\u0902? \u0915\u0943\u092a\u092f\u093e 1 \u0938\u0947 9 \u092e\u0939\u0940\u0928\u0947 \u0915\u0947 \u092c\u0940\u091a \u092e\u0947\u0902 \u092c\u0924\u093e\u0907\u090f\u0964",
                    "\u0906\u092a\u0915\u094b \u0917\u0930\u094d\u092d\u0935\u0924\u0940 \u0939\u0941\u090f \u0915\u093f\u0924\u0928\u0947 \u092e\u0939\u0940\u0928\u0947 \u0939\u0941\u090f \u0939\u0948\u0902? \u0915\u0943\u092a\u092f\u093e 1 \u0938\u0947 9 \u092e\u0939\u0940\u0928\u0947 \u0915\u0947 \u092c\u0940\u091a \u092e\u0947\u0902 \u092c\u0924\u093e\u0907\u090f\u0964"
                ],
                "exhausted_action": "end_call"
            }
        },
        {
            "name": "child_name",
            "label": "child_name",
            "type": "string",
            "required": false,
            "script": "\u0906\u092a\u0915\u0947 \u092c\u091a\u094d\u091a\u0947 \u0915\u093e \u0928\u093e\u092e \u0915\u094d\u092f\u093e \u0939\u0948?",
            "branch_id": "branch_not_pregnant",
            "retry_config": {
                "allow_retries": true,
                "until_answered": false,
                "max_retries": 1,
                "retry_messages": [
                    "\u0915\u0943\u092a\u092f\u093e \u0905\u092a\u0928\u0947 \u092c\u091a\u094d\u091a\u0947 \u0915\u093e \u092a\u0942\u0930\u093e \u0928\u093e\u092e \u092c\u0924\u093e\u0907\u090f\u0964"
                ],
                "exhausted_action": "end_call"
            }
        },
        {
            "name": "child_dob",
            "label": "child_dob",
            "type": "date",
            "required": false,
            "script": "\u0906\u092a\u0915\u0947 \u092c\u091a\u094d\u091a\u0947 \u0915\u0940 \u091c\u0928\u094d\u092e \u0924\u093e\u0930\u0940\u0916 \u092c\u0924\u093e\u090f\u0902",
            "branch_id": "branch_not_pregnant",
            "validation": {
                "type": "date_today_or_past",
                "rules": {}
            },
            "retry_config": {
                "allow_retries": true,
                "until_answered": false,
                "max_retries": 2,
                "retry_messages": [
                    "\u0915\u0943\u092a\u092f\u093e \u0905\u092a\u0928\u0947 \u092c\u091a\u094d\u091a\u0947 \u0915\u0940 \u091c\u0928\u094d\u092e \u0924\u093e\u0930\u0940\u0916 \u0926\u093f\u0928, \u092e\u0939\u0940\u0928\u093e \u0914\u0930 \u0935\u0930\u094d\u0937 \u0915\u0947 \u0915\u094d\u0930\u092e \u092e\u0947\u0902 \u092c\u0924\u093e\u090f\u0902 \u091c\u0948\u0938\u0947 \u0915\u093f 15 \u0905\u092a\u094d\u0930\u0948\u0932 2025\u0964",
                    "\u0915\u0943\u092a\u092f\u093e \u0905\u092a\u0928\u0947 \u092c\u091a\u094d\u091a\u0947 \u0915\u0940 \u091c\u0928\u094d\u092e \u0924\u093e\u0930\u0940\u0916 \u0926\u093f\u0928, \u092e\u0939\u0940\u0928\u093e \u0914\u0930 \u0935\u0930\u094d\u0937 \u0915\u0947 \u0915\u094d\u0930\u092e \u092e\u0947\u0902 \u092c\u0924\u093e\u090f\u0902 \u091c\u0948\u0938\u0947 \u0915\u093f 15 \u0905\u092a\u094d\u0930\u0948\u0932 2025\u0964"
                ],
                "exhausted_action": "end_call"
            }
        },
        {
            "name": "is_phone_linked_to_anganwadi",
            "label": "is_phone_linked_to_anganwadi",
            "type": "boolean",
            "required": false,
            "advanced_instructions": "Before asking this question, say the following informational script EXACTLY and then ask the question: \"\u0905\u092c \u0915\u0947\u0935\u0932 4 \u0938\u0935\u093e\u0932 \u092c\u093e\u0915\u0940 \u0939\u0948\u0902\u0964\"",
            "script": "\u0915\u094d\u092f\u093e \u092f\u0939 \u0928\u0902\u092c\u0930 \u0906\u0902\u0917\u0928\u0935\u093e\u0921\u093c\u0940 \u0938\u0947 \u091c\u0941\u0921\u093c\u093e \u0939\u0941\u0906 \u0939\u0948?",
            "retry_config": {
                "allow_retries": true,
                "until_answered": false,
                "max_retries": 2,
                "retry_messages": [
                    "\u0915\u0943\u092a\u092f\u093e \u0906\u092a\u0915\u0947 \u092b\u094b\u0928 \u092a\u0930 \u0906\u090f \u0939\u0941\u090f SMS \u092e\u0947\u0902 6 \u0905\u0902\u0915\u094b\u0902 \u0915\u093e \u0915\u094b\u0921 \u092c\u0924\u093e\u0907\u090f\u0964",
                    "\u0915\u0943\u092a\u092f\u093e \u0906\u092a\u0915\u0947 \u092b\u094b\u0928 \u092a\u0930 \u0906\u090f \u0939\u0941\u090f SMS \u092e\u0947\u0902 6 \u0905\u0902\u0915\u094b\u0902 \u0915\u093e \u0915\u094b\u0921 \u092c\u0924\u093e\u0907\u090f\u0964 \u0905\u0917\u0930 \u0906\u092a\u0915\u094b \u0928\u0939\u0940\u0902 \u092a\u0924\u093e \u0939\u0948, \u0924\u094b \\\"\u0928\u0939\u0940\u0902 \u092a\u0924\u093e\\\" \u092c\u094b\u0932\u093f\u090f\u0964"
                ],
                "exhausted_action": "skip"
            }
        },
        {
            "name": "anganwadi_linked_phone_number",
            "label": "anganwadi_linked_phone_number",
            "type": "number",
            "number_format": "integer",
            "required": false,
            "script": "\u0915\u0943\u092a\u092f\u093e \u0906\u0902\u0917\u0928\u0935\u093e\u0921\u093c\u0940 \u0938\u0947 \u091c\u0941\u0921\u093c\u093e \u0939\u0941\u0906 \u0905\u092a\u0928\u093e \u092b\u094b\u0928 \u0928\u0902\u092c\u0930 \u092c\u0924\u093e\u090f\u0902\u0964",
            "branch_id": "branch_phone_not_linked",
            "validation": {
                "type": "phone_number",
                "rules": {
                    "digits": 10
                }
            },
            "retry_config": {
                "allow_retries": true,
                "until_answered": false,
                "max_retries": 2,
                "retry_messages": [
                    "\u0906\u0902\u0917\u0928\u0935\u093e\u0921\u093c\u0940 \u092e\u0947\u0902 \u091c\u0941\u0921\u093c\u093e \u0939\u0941\u0906 \u0906\u092a\u0915\u093e 10 \u0905\u0902\u0915\u094b\u0902 \u0915\u093e \u092b\u094b\u0928 \u0928\u0902\u092c\u0930 \u092c\u0924\u093e\u0907\u090f\u0964",
                    "\u0906\u0902\u0917\u0928\u0935\u093e\u0921\u093c\u0940 \u092e\u0947\u0902 \u091c\u0941\u0921\u093c\u093e \u0939\u0941\u0906 \u0906\u092a\u0915\u093e 10 \u0905\u0902\u0915\u094b\u0902 \u0915\u093e \u092b\u094b\u0928 \u0928\u0902\u092c\u0930 \u092c\u0924\u093e\u0907\u090f\u0964 \u0905\u0917\u0930 \u0928\u0939\u0940\u0902 \u092a\u0924\u093e \u0939\u0948, \u0924\u094b \\\"\u0928\u0939\u0940\u0902 \u092a\u0924\u093e\\\" \u0910\u0938\u093e \u092c\u094b\u0932\u093f\u090f\u0964"
                ],
                "exhausted_action": "skip"
            }
        },
        {
            "name": "whatsapp_linked_to_calling_number",
            "label": "whatsapp_linked_to_calling_number",
            "type": "boolean",
            "required": false,
            "script": "\u0906\u092a\u0928\u0947 \u091c\u093f\u0938 \u0928\u0902\u092c\u0930 \u0938\u0947 \u0915\u0949\u0932 \u0915\u093f\u092f\u093e \u0939\u0948, \u0915\u094d\u092f\u093e \u0935\u0939\u0940 \u0928\u0902\u092c\u0930 \u0906\u092a\u0915\u0947 \u0935\u094d\u0939\u093e\u091f\u094d\u0938\u0910\u092a \u0938\u0947 \u092d\u0940 \u091c\u0941\u0921\u093c\u093e \u0939\u0941\u0906 \u0939\u0948?",
            "retry_config": {
                "allow_retries": true,
                "until_answered": false,
                "max_retries": 2,
                "retry_messages": [
                    "\u0906\u092a\u0928\u0947 \u091c\u093f\u0938 \u0928\u0902\u092c\u0930 \u0938\u0947 \u0915\u0949\u0932 \u0915\u093f\u092f\u093e \u0939\u0948, \u0915\u094d\u092f\u093e \u0935\u0939 \u0935\u094d\u0939\u093e\u091f\u094d\u0938\u090f\u092a \u092e\u0947\u0902 \u092d\u0940 \u0926\u0930\u094d\u091c \u0939\u0948? \u0915\u0943\u092a\u092f\u093e \u0939\u093e\u0901 \u092f\u093e \u0928\u093e \u092e\u0947\u0902 \u091c\u0935\u093e\u092c \u0926\u0947\u0902\u0964",
                    "\u0906\u092a\u0928\u0947 \u091c\u093f\u0938 \u0928\u0902\u092c\u0930 \u0938\u0947 \u0915\u0949\u0932 \u0915\u093f\u092f\u093e \u0939\u0948, \u0915\u094d\u092f\u093e \u0935\u0939 \u0935\u094d\u0939\u093e\u091f\u094d\u0938\u090f\u092a \u092e\u0947\u0902 \u092d\u0940 \u0926\u0930\u094d\u091c \u0939\u0948? \u0905\u0917\u0930 \u0928\u0939\u0940\u0902 \u092a\u0924\u093e \u0939\u0948, \u0924\u094b \\\"\u0928\u0939\u0940\u0902 \u092a\u0924\u093e\\\" \u092c\u094b\u0932\u093f\u090f\u0964"
                ],
                "exhausted_action": "skip"
            }
        },
        {
            "name": "whatsapp_number",
            "label": "whatsapp_number",
            "type": "number",
            "number_format": "integer",
            "required": false,
            "script": "\u0915\u0943\u092a\u092f\u093e \u0905\u092a\u0928\u093e \u0935\u094d\u0939\u093e\u091f\u094d\u0938\u0910\u092a \u0928\u0902\u092c\u0930 \u092c\u0924\u093e\u0907\u090f\u0964",
            "branch_id": "branch_whatsapp_not_linked",
            "validation": {
                "type": "phone_number",
                "rules": {
                    "digits": 10
                }
            },
            "retry_config": {
                "allow_retries": true,
                "until_answered": false,
                "max_retries": 2,
                "retry_messages": [
                    "\u0935\u094d\u0939\u093e\u091f\u094d\u0938\u090f\u092a \u092e\u0947\u0902 \u091c\u0941\u0921\u093c\u093e \u0939\u0941\u0906 \u0906\u092a\u0915\u093e 10 \u0905\u0902\u0915\u094b\u0902 \u0915\u093e \u092b\u094b\u0928 \u0928\u0902\u092c\u0930 \u092c\u0924\u093e\u0907\u090f\u0964",
                    "\u0935\u094d\u0939\u093e\u091f\u094d\u0938\u090f\u092a \u092e\u0947\u0902 \u091c\u0941\u0921\u093c\u093e \u0939\u0941\u0906 \u0906\u092a\u0915\u093e 10 \u0905\u0902\u0915\u094b\u0902 \u0915\u093e \u092b\u094b\u0928 \u0928\u0902\u092c\u0930 \u092c\u0924\u093e\u0907\u090f \u0905\u0917\u0930 \u0928\u0939\u0940\u0902 \u092a\u0924\u093e \u0939\u0948, \u0924\u094b \\\"\u0928\u0939\u0940\u0902 \u092a\u0924\u093e\\\" \u0910\u0938\u093e \u092c\u094b\u0932\u093f\u090f\u0964"
                ],
                "exhausted_action": "skip"
            }
        },
        {
            "name": "last_4_digits_of_aadhar",
            "label": "last_4_digits_of_aadhar",
            "type": "number",
            "number_format": "integer",
            "required": false,
            "advanced_instructions": "Before asking this question, say the following informational script EXACTLY: \"\u0905\u092c \u091c\u094b \u0938\u0935\u093e\u0932 \u0906\u090f\u0917\u093e, \u0935\u0939 \u0906\u0916\u093c\u093f\u0930\u0940 \u0938\u0935\u093e\u0932 \u0939\u094b\u0917\u093e\u0964\"",
            "script": "\u0905\u092a\u0928\u0947 \u0906\u0927\u093e\u0930 \u0915\u0947 \u0906\u0916\u093f\u0930\u0940 4 \u0928\u0902\u092c\u0930 \u092c\u0924\u093e\u0907\u090f\u0964",
            "validation": {
                "type": "exact_num_digits",
                "rules": {
                    "digits": 4
                }
            },
            "retry_config": {
                "allow_retries": true,
                "until_answered": false,
                "max_retries": 2,
                "retry_messages": [
                    "\u0915\u0943\u092a\u092f\u093e \u0905\u092a\u0928\u0947 \u0906\u0927\u093e\u0930 \u0915\u093e\u0930\u094d\u0921 \u0915\u0947 \u0906\u0916\u093f\u0930\u0940 4 \u0928\u0902\u092c\u0930 \u092c\u0924\u093e\u0907\u090f\u0964",
                    "\u0915\u0943\u092a\u092f\u093e \u0905\u092a\u0928\u0947 \u0906\u0927\u093e\u0930 \u0915\u093e\u0930\u094d\u0921 \u0915\u0947 \u0906\u0916\u093f\u0930\u0940 4 \u0928\u0902\u092c\u0930 \u092c\u0924\u093e\u0907\u090f\u0964 \u0905\u0917\u0930 \u0928\u0939\u0940\u0902 \u092a\u0924\u093e \u0939\u0948, \u0924\u094b \\\"\u0928\u0939\u0940\u0902 \u092a\u0924\u093e\\\" \u092c\u094b\u0932\u093f\u090f\u0964"
                ],
                "exhausted_action": "skip"
            }
        }
    ],
    "branches": [
        {
            "id": "branch_pregnant",
            "condition": {
                "field": "pregnant_or_not",
                "equals": true
            }
        },
        {
            "id": "branch_not_pregnant",
            "condition": {
                "field": "pregnant_or_not",
                "equals": false
            }
        },
        {
            "id": "branch_phone_not_linked",
            "condition": {
                "field": "is_phone_linked_to_anganwadi",
                "equals": false
            }
        },
        {
            "id": "branch_whatsapp_not_linked",
            "condition": {
                "field": "whatsapp_linked_to_calling_number",
                "equals": false
            }
        }
    ],
    "context": "This is a health survey call from Swasth Kadam to collect basic information from beneficiaries registered at Anganwadi centres. The agent is a health worker and the user is a pregnant mother or a mother with a young child. The agent is a health worker and the user is a pregnant mother. The agent is a health worker and the user is a pregnant mother. The agent is a health worker and the user is a pregnant mother. The agent is a health worker and the user is a pregnant mother.",
    "validation_rules": "- 'name': Must be at least 2 characters long.\n- 'gestational_age_months': Must be between 1 and 9.\n- 'child_dob': Must be a valid DD-MM-YYYY date that is today or earlier.\n- 'anganwadi_linked_phone_number': Must be a valid phone number with exactly 10 digits.\n- 'whatsapp_number': Must be a valid phone number with exactly 10 digits.\n- 'last_4_digits_of_aadhar': Must be exactly 4 digits.",
    "instructions": "IMPORTANT RETRY AND CALL-END RULES:\n\n- When retrying, use the EXACT retry script text specified for that retry attempt. Do NOT paraphrase.\n- Some questions have an informational script that must be spoken BEFORE asking that question. Follow those instructions exactly.\n- A brief hearing-only acknowledgement is already spoken before your reply \u2014 do NOT start with any acknowledgement or phrase implying you recorded, noted, or registered their answer (e.g. avoid '\u0926\u0930\u094d\u091c \u0915\u0930 \u0932\u093f\u092f\u093e', '\u091c\u093e\u0928\u0915\u093e\u0930\u0940 \u092e\u093f\u0932 \u0917\u0908', '\u0938\u092e\u091d \u0917\u0908'). Go straight to the next question Script, retry, or correction.\n- If the user explicitly asks you to repeat the question (e.g., 'Phir se bolo', 'Repeat karo', 'Dobara bolo'), repeat the current question exactly and do NOT count it as a retry attempt. If the user says they did not understand (e.g., 'Samajh nahi aaya', 'Kya bola', 'Sunai nahi diya'), treat it as a failed attempt and proceed according to the retry rules.",
    "agent_speaks_first": true,
    "ask_questions_one_by_one": true,
    "status": "published",
    "agent_persona": "health worker",
    "user_persona": "pregnant mother",
    "language": "hindi",
    "scripts": {
        "intro": "\u0928\u092e\u0938\u094d\u0924\u0947!",
        "outro": "\u091c\u093e\u0928\u0915\u093e\u0930\u0940 \u0926\u0947\u0928\u0947 \u0915\u0947 \u0932\u093f\u090f \u0927\u0928\u094d\u092f\u0935\u093e\u0926\u0964 \u0939\u092e \u0906\u092a\u0915\u094b \u091c\u0932\u094d\u0926 \u0939\u0940 \u0939\u092e\u093e\u0930\u0940 \u0938\u0947\u0935\u093e \u0938\u0947 \u091c\u094b\u0921\u093c\u0947\u0902\u0917\u0947\u0964",
        "outro_incomplete": "\u0915\u094d\u0937\u092e\u093e \u0915\u0940\u091c\u093f\u090f, \u0939\u092e \u0907\u0938 \u0938\u092e\u092f \u0906\u092a\u0915\u0940 \u091c\u093e\u0928\u0915\u093e\u0930\u0940 \u0926\u0930\u094d\u091c \u0928\u0939\u0940\u0902 \u0915\u0930 \u092a\u093e\u090f, \u0907\u0938\u0932\u093f\u090f \u0905\u092d\u0940 \u0915\u0949\u0932 \u0938\u092e\u093e\u092a\u094d\u0924 \u0915\u0940 \u091c\u093e \u0930\u0939\u0940 \u0939\u0948\u0964 \u0927\u0928\u094d\u092f\u0935\u093e\u0926\u0964"
    }
}
```

### 3. The agent URL Calibrate connects to
Calibrate uses the pipecat non-telephony WebSocket endpoint:
```
ws://localhost:7860/ws-client/<agent-id>
```
`<agent-id>` is a folder name under `server/db/agents/`. The experiments use the
**`swasth-kadam-v5`** agent (a Hindi child-enrollment form), so the URL is:
```
ws://localhost:7860/ws-client/swasth-kadam-v5
```
This exact URL is already set as `agent_url` in the example configs. To use a
different agent, drop its `config.json` under `server/db/agents/<its-id>/` and
point `agent_url` at `ws://localhost:7860/ws-client/<its-id>`.

### 4. Where the agent stores what it captured (ground truth)
Every call creates a conversation folder on the agent side:
```
server/db/agents/swasth-kadam-v5/conversations/<conversation-id>/
├── data.json        # the FINAL filled form (form_data + form_status)  ← ground truth
├── transcript.json  # the agent's view of the conversation
├── recording.wav
├── metrics.json
└── events.json / conversation.log / turns/
```
`data.json → form_data` is the authoritative "what actually landed in the form",
used later to judge accuracy (see Part E).

---

## Part B — Set up Calibrate (this branch)

### 1. Clone, check out the branch, install
```bash
git clone git@github.com:ARTPARK-SAHAI-ORG/calibrate.git
cd calibrate
git checkout claude/festive-rosalind-92d0d6
uv sync --extra dev
```

### 2. Provider keys — `.env` in the calibrate repo root
Create a `.env` file at the **root of the calibrate repo** (`calibrate/.env`). The
**simulated caller** side needs these (the agent uses its own keys, Part A):
| Key | Used for |
|---|---|
| `OPENAI_API_KEY` | simulated-caller LLM + the evaluator judges |
| `OPENROUTER_API_KEY` | evaluator judge models routed via OpenRouter |
| `ELEVENLABS_API_KEY` | simulated-caller voice (English/Hindi) |
| `GOOGLE_APPLICATION_CREDENTIALS` | simulated-caller voice (Kannada, Google Chirp3-HD) |
| `HF_TOKEN` (or `huggingface-cli login`) | pulling Vaani speaker clips (Part B.3) |

**If you don't know what to put in this file, ask Aman.**

### 3. Build the noise assets  ⚠️ required, not in git
The audio assets are **gitignored** (licensing), so you must build them once. They
live under `calibrate_agent/agent/assets/noise/` and are generated from raw clips in
`data/`:

```
data/env_raw/                      # environmental sounds (ESC-50, 16-bit)
├── rain.wav wind.wav engine.wav vacuum_cleaner.wav train.wav siren.wav
├── crying_baby.wav footsteps.wav keyboard_typing.wav laughing.wav
├── dog/00.wav … 07.wav           # multiple samples (scattered, not looped)
└── car_horn/00.wav … 07.wav
data/vaani_raw/{english,hindi,kannada}/*.wav   # Vaani speaker clips, 16 kHz mono
```
- **Environmental clips**: the 13 ESC-50 classes above (ESC-50 is public; note it is
  CC BY-NC, so treat as local-only until CC0-re-sourced before shipping).
- **Speaker clips**: from the gated **ARTPARK-IISc/Vaani** HF dataset (CC BY 4.0) —
  accept its terms + `huggingface-cli login`, then pull ~25–40 clips per language
  (english/hindi/kannada), filtering rows by the `language` field.

Then bundle them (resamples to 16 kHz mono):
```bash
uv run python -c "from calibrate_agent.agent.noise.assets import prepare_assets; prepare_assets()"
```

**Sanity check (no keys/assets needed):**
```bash
uv run pytest tests/agent/test_noise_*.py -q      # 57 tests, synthetic fixtures
```

---

## Part C — The config

A voice-sim config is one JSON with `personas`, `scenarios`, `evaluators`, `settings`,
and `agent_url`. **Noise is a per-persona setting** — add a `noise` block inside each
persona:

```jsonc
"noise": {
  "mode": "fixed",              // off | fixed | random | mixture
  "environment": "busy_street", // single sound, a scene, ["list"], or null
  "people": "light",            // none | single | light | medium | heavy  (chatter, language-matched)
  "loudness": "moderate"        // faint | moderate | loud | harsh   (louder = harder)
}
```
Omit `noise` on a persona for a clean control. Difficulty rises with **loudness**
(faint→harsh) and **people density** (crowds mask speech more than steady sounds).

The two experiment configs live at
`examples/agent/simulation/sample_voice_noise_{easy,hard}.json` and are reproduced
below. **Easy** = env/people/mixed at faint→moderate; **Hard** = the same at
loud→harsh; each ends with a clean baseline.

### `sample_voice_noise_easy.json`
```json
{
  "agent_url": "ws://localhost:7860/ws-client/swasth-kadam-v5",
  "personas": [
    { "label": "easy L1 · env · rain · faint", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. a little soft rain outside.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none", "noise": { "mode": "fixed", "environment": "rain", "people": "none", "loudness": "faint" } },
    { "label": "easy L2 · env · wind · faint", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. a soft breeze outside.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none", "noise": { "mode": "fixed", "environment": "wind", "people": "none", "loudness": "faint" } },
    { "label": "easy L3 · people · single · faint", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. one person talking quietly nearby.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none", "noise": { "mode": "fixed", "environment": null, "people": "single", "loudness": "faint" } },
    { "label": "easy L4 · env · busy street · moderate", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. standing near a moderately busy street.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none", "noise": { "mode": "fixed", "environment": "busy_street", "people": "none", "loudness": "moderate" } },
    { "label": "easy L5 · people · light · moderate", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. a couple of people chatting nearby.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none", "noise": { "mode": "fixed", "environment": null, "people": "light", "loudness": "moderate" } },
    { "label": "easy L6 · mixed · street+light · moderate", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. a busy street with a few people around.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none", "noise": { "mode": "fixed", "environment": "busy_street", "people": "light", "loudness": "moderate" } },
    { "label": "easy · baseline · clean", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. you are at home in a quiet room.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none" }
  ],
  "scenarios": [
    { "name": "straightforward enrollment", "description": "Complete the Swasth Kadham signup for your child. Answer clearly right away. Facts: full name Priya Mehta; district Pune; Anganwadi Lakshmi Nagar Kendra; not pregnant; child Aarav Mehta, born 22 July 2023; this number is linked to your Anganwadi; WhatsApp on this number; last 4 Aadhaar digits 4827." }
  ],
  "evaluators": [
    { "id": "form-complete-id", "name": "form_completed", "system_prompt": "Mark True if the agent collected the key enrollment fields (name, district, pregnancy status, child name+DOB, phone/WhatsApp linkage, last 4 Aadhaar digits) and gave a closing acknowledgment.", "judge_model": "openai/gpt-4.1" },
    { "id": "patient-guidance-id", "name": "patient_guidance", "system_prompt": "Mark True if the agent was polite, asked questions one at a time, and patiently drew out information under noise.", "judge_model": "openai/gpt-4.1" }
  ],
  "settings": { "agent_speaks_first": true, "max_turns": 50 }
}
```

### `sample_voice_noise_hard.json`
```json
{
  "agent_url": "ws://localhost:7860/ws-client/swasth-kadam-v5",
  "personas": [
    { "label": "hard L1 · env · vehicle · loud", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. on a moving bus, engine loud.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none", "noise": { "mode": "fixed", "environment": "vehicle", "people": "none", "loudness": "loud" } },
    { "label": "hard L2 · people · medium · loud", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. in a cafe with several people talking.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none", "noise": { "mode": "fixed", "environment": null, "people": "medium", "loudness": "loud" } },
    { "label": "hard L3 · mixed · station+medium · loud", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. at a crowded, loud railway station.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none", "noise": { "mode": "fixed", "environment": "railway_station", "people": "medium", "loudness": "loud" } },
    { "label": "hard L4 · env · vacuum · harsh", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. at home with a vacuum running right nearby.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none", "noise": { "mode": "fixed", "environment": "vacuum", "people": "none", "loudness": "harsh" } },
    { "label": "hard L5 · people · heavy · harsh", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. in a packed, very noisy crowd.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none", "noise": { "mode": "fixed", "environment": null, "people": "heavy", "loudness": "harsh" } },
    { "label": "hard L6 · mixed · rainy+heavy · harsh", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. on a rainy street in a heavy crowd.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none", "noise": { "mode": "fixed", "environment": "rainy_street", "people": "heavy", "loudness": "harsh" } },
    { "label": "hard · baseline · clean", "characteristics": "You are a Hindi-speaking mother enrolling your child in Swasth Kadham. You answer politely, one question at a time. you are at home in a quiet room.", "gender": "female", "language": "hindi", "interruption_sensitivity": "none" }
  ],
  "scenarios": [
    { "name": "straightforward enrollment", "description": "Complete the Swasth Kadham signup for your child. Answer clearly right away. Facts: full name Priya Mehta; district Pune; Anganwadi Lakshmi Nagar Kendra; not pregnant; child Aarav Mehta, born 22 July 2023; this number is linked to your Anganwadi; WhatsApp on this number; last 4 Aadhaar digits 4827." }
  ],
  "evaluators": [
    { "id": "form-complete-id", "name": "form_completed", "system_prompt": "Mark True if the agent collected the key enrollment fields (name, district, pregnancy status, child name+DOB, phone/WhatsApp linkage, last 4 Aadhaar digits) and gave a closing acknowledgment.", "judge_model": "openai/gpt-4.1" },
    { "id": "patient-guidance-id", "name": "patient_guidance", "system_prompt": "Mark True if the agent was polite, asked questions one at a time, and patiently drew out information under noise.", "judge_model": "openai/gpt-4.1" }
  ],
  "settings": { "agent_speaks_first": true, "max_turns": 50 }
}
```

---

## Part D — Run the experiments

With the **agent running** (Part A) and **assets built** (Part B):

```bash
# from the calibrate repo root, on branch claude/festive-rosalind-92d0d6
uv run calibrate-agent simulations --type voice \
  -c examples/agent/simulation/sample_voice_noise_easy.json -o ./out/easy --parallel 4

uv run calibrate-agent simulations --type voice \
  -c examples/agent/simulation/sample_voice_noise_hard.json -o ./out/hard --parallel 4
```
- `-o <dir>` — a **different output folder per run**.
- `--parallel N` — run N calls at once (each is a real voice conversation; 4 is a good default). Each persona × scenario is one call.
- Each call is a full Hindi conversation, so a 7-persona batch takes a few minutes.

**If you want to match each call to its agent-side `data.json` (Part E), run
`--parallel 1`** — sequential calls create agent conversations in persona order, which
is the only reliable way to line them up (parallel + identical scenarios can't be
matched).

Other ready-made configs in the same folder: `sample_voice_noise_matrix.json`
(type × difficulty grid), `_progressive.json` (one smooth ramp), `_showcase.json`
(demonstrates random/mixture modes).

---

## Part E — Reading the output

### Run-level (in `-o` dir)
```
out/easy/
├── results.csv     # one row per sim: form_completed, patient_guidance, stt_llm_judge_score, e2e_latency
├── metrics.json    # aggregated latency/STT metrics
└── simulation_persona_<i>_scenario_<j>/   # one folder per call
```

### Per-simulation folder
```
simulation_persona_1_scenario_1/
├── conversation.wav        # what the agent HEARD — caller speech + background noise
├── clean_conversation.wav  # the caller's speech WITHOUT noise (reference)
├── clean_*.wav             # per-turn clean copies
├── noise_track.wav         # the exact looping background that was mixed in (afplay to hear it)
├── transcript.json         # the conversation (agent messages + caller turns)
├── stt_results.csv         # per-utterance: reference (what caller said) vs prediction (what agent heard) + score
├── stt_outputs.json        # raw agent transcriptions
├── evaluation_results.csv  # the judge scores (form_completed, patient_guidance)
├── metrics.json            # per-call latency/STT
├── audios/                 # per-turn wavs
└── tool_calls.json, config.json, logs, results.log
```
Quick listen — noisy vs clean:
```bash
afplay out/easy/simulation_persona_5_scenario_1/conversation.wav        # agent-heard (noisy)
afplay out/easy/simulation_persona_5_scenario_1/clean_conversation.wav  # clean
afplay out/easy/simulation_persona_5_scenario_1/noise_track.wav         # the noise loop alone
```
