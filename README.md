# AgentLoop

An open-source simulation and testing framework for voice agents.

With AgentLoop, you can move from a slow, manual testing to a fast, automated, and repeatable testing process:

- Evaluate different components of your voice agents across multiple vendors in isolation
- Create comprehensive test suites that verify both conversational responses and tool usage
- Simulate entire conversations spanning thousands of scenarios across multiple user personas

AgentLoop is built on top of [pipecat](https://github.com/pipecat-ai/pipecat), a framework for building voice agents.

## Setup

We use `uv` to manage the dependencies. Make sure you have it installed on your system.

Create the virtual environment and install the dependencies.

```bash
uv sync --frozen
```

## Talk to the agent

`src/agent/run.py` spins up the fully interactive STT → LLM → TTS pipeline so you can speak with the agent in real time.

1. Copy `src/agent/.env.example` to `src/agent/.env` and provide the API keys for the STT/LLM/TTS providers you want to exercise (the defaults in `run.py` use OpenAI STT + LLM and Google TTS).
2. Pick or edit a config file. `src/agent/examples/run/config.json` is a ready-to-use config file.
3. Start the runner:

```bash
cd src/agent
uv run python run.py -c examples/run/config.json -o ./out/run
```

Once you run it, you can open `http://localhost:7860/client/` in your browser. Click **Connect**, and begin talking to the agent through your browser. The client UI streams the conversation transcript in real time, while the **Metrics** tab mirrors the live latency statistics reported by the pipeline as shwon in the images below:

![Conversation example](images/run_conversation.png)

![Metrics example](images/run_metrics.png)

Every user/bot audio turn is persisted inside `<output_dir>/audios` (see a sample output folder for the exact layout). The terminal also logs each transcript message and every function/tool call issued by the LLM so you can audit the interaction without leaving your shell.

You can checkout `src/agent/examples/run/sample_output` as a sample output of the agent conversation which contains all the audio turns and logs from the conversation.

## Speech To Text (STT)

To evaluate different STT providers, first make sure to organize the input data in the following structure:

```bash
├── /path/to/data
│   └── stt.csv
│   └── audios/wav
│       └── audio_1.wav
│       └── audio_2.wav
│   └── audios/pcm16
│       └── audio_1.wav
│       └── audio_2.wav
```

`stt.csv` should have the following format:

```csv
id,text
audio_1,"Hi"
audio_2,"Madam, my name is Geeta Shankar"
```

Store the audios in both `wav` and `pcm16` formats as different providers support different formats. The evaluation script will automatically select the right format for the provider you choose to evaluate.

Copy `src/stt/.env.example` to `src/stt/.env` and fill in the API keys for the providers you want to evaluate.

```bash
cd src/stt
uv run python eval.py --input-dir /path/to/data -o /path/to/output -d -p <provider> -l <language>
```

You can use the sample inputs provided in [`src/stt/examples/sample_input`](src/stt/examples/sample_input) to test the evaluation script.

```bash
cd src/stt
uv run python eval.py --input-dir samples -o ./out -d -p sarvam -l english
```

Sample output:

```bash
Created new user transcript
Starting new audio streaming: /path/to/audio_1.wav
Finished streaming audio: /path/to/audio_1.wav
appending to last user transcript: <transcript>.
--------------------------------
Created new user transcript
Starting new audio streaming: /path/to/audio_2.wav
Finished streaming audio: /path/to/audio_2.wav
appending to last user transcript: <transcript_part_1>,
appending to last user transcript: <transcript_part_2>,
```

The output of the evaluation script will be saved in the output directory.

```bash
/path/to/output
├── sarvam_english
│   ├── logs
│   ├── results.log
│   ├── results.csv
│   └── metrics.json
```

`results.csv` will have the following columns:

```csv
id,gt,pred,wer,string_similarity,llm_judge_score,llm_judge_reasoning
3_1_english_baseline,"Please write Rekha Kumari, sister.", Please write Reha Kumari's sister.,0.4,0.927536231884058,False,"The source says 'Rekha Kumari, sister' which indicates the name is 'Rekha Kumari' and she is a sister. The transcription says 'Reha Kumari's sister', which changes the name to 'Reha Kumari' and refers to her sister, not Rekha Kumari herself. The name is different ('Rekha' vs 'Reha') and the relationship is also changed (from identifying Rekha Kumari as the sister to referring to the sister of Reha Kumari). Therefore, the values do not match."
```

The definition of all the metrics we compute are stored in [`src/stt/metrics.py`](src/stt/metrics.py).

`metrics.json` will have the following format:

```json
{
  "wer": 0.12962962962962962, // mean of the Word Error Rate (WER) across all audio files
  "string_similarity": 0.8792465033551621, // mean of the string similarity score across all audio files
  "llm_judge_score": 1.0, // mean of the LLM Judge score across all audio files
  "metrics": [
    // latency metrics for each audio file
    {
      "metric_name": "ttfb", // Time to First Byte in seconds
      "processor": "SarvamSTTService#0",
      "mean": 2.3087445222414456,
      "std": 2.4340144350359867,
      "values": [
        // values for each audio file
        0.48694515228271484, 0.006701946258544922, 0.4470040798187256, 2.5300347805023193,
        0.2064838409423828, 6.574702978134155, 3.0182559490203857, 2.9680559635162354,
        7.94110107421875, 0.08572530746459961, 2.0372867584228516, 0.4102351665496826,
        3.3011457920074463
      ]
    },
    {
      "metric_name": "processing_time", // Time taken by the service to respond in seconds
      "processor": "SarvamSTTService#0",
      "mean": 2.3089125706599307,
      "std": 2.4341351775837228,
      "values": [
        // values for each audio file
        0.48702311515808105, 0.006762981414794922, 0.4470367431640625, 2.530163049697876,
        0.2065589427947998, 6.575268745422363, 3.0185441970825195, 2.9680910110473633,
        7.941352128982544, 0.08578085899353027, 2.0373899936676025, 0.4102756977081299,
        3.3016159534454346
      ]
    }
  ]
}
```

`logs` contain the full logs of the evaluation script including all the pipecat logs whereas `results.log` contains only the terminal output of the evaluation script as shown in the sample output above.

You can checkout [`src/stt/examples/sample_output`](src/stt/examples/sample_output) to see a sample output of the evaluation script.

For more details, run `uv run python eval.py -h`.

```bash
usage: eval.py [-h] [-p {deepgram,deepgram-flux,openai,cartesia,smallest,groq,google,sarvam}] [-l {english,hindi}] -i
               INPUT_DIR [-o OUTPUT_DIR] [-d]

options:
  -h, --help            show this help message and exit
  -p {deepgram,deepgram-flux,openai,cartesia,smallest,groq,google,sarvam}, --provider {deepgram,deepgram-flux,openai,cartesia,smallest,groq,google,sarvam}
                        STT provider to use for evaluation
  -l {english,hindi}, --language {english,hindi}
                        Language of the audio files
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Path to the input directory containing the audio files and gt.json
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Path to the output directory to save the results
  -d, --debug           Run the evaluation on the first 5 audio files
```

After you have multiple provider runs under `/path/to/output`, you can summarize accuracy and latency in one shot:

```bash
cd src/stt
uv run python leaderboard.py -o /path/to/output -s ./leaderboards
```

The script scans each run directory, reads `metrics.json` and `results.csv`, then writes `stt_leaderboard.xlsx` plus `all_metrics_by_run.png` inside the save directory so you can compare providers side-by-side (`src/stt/leaderboard.py`).

You can checkout [`src/stt/examples/leaderboard`](src/stt/examples/leaderboard) to see a sample output of the leaderboard script.

![Leaderboard example](src/stt/examples/leaderboard/all_metrics_by_run.png)

## Text To Speech (TTS)

To evaluate different TTS providers, first prepare an input CSV file with the following structure:

```csv
text
hello world
this is a test
```

The CSV should have a single column `text` containing the text strings you want to synthesize into speech.

Copy `src/tts/.env.example` to `src/tts/.env` and fill in the API keys for the providers you want to evaluate.

```bash
cd src/tts
uv run python eval.py --input /path/to/input.csv -o /path/to/output -p <provider> -l <language>
```

You can use the sample inputs provided in [`src/tts/examples/sample.csv`](src/tts/examples/sample.csv) to test the evaluation script.

```bash
cd src/tts
uv run python eval.py --input examples/sample.csv -o ./out -p smallest -l english
```

The output of the evaluation script will be saved in the output directory.

```bash
/path/to/output/provider
├── audios
│   ├── 1.wav
│   ├── 2.wav
├── logs
├── results.csv
└── metrics.json
```

`results.csv` will have the following columns:

```csv
text,audio_path
hello world,./out/audios/1.wav
this is a test,./out/audios/2.wav
```

`metrics.json` will have the following format:

```json
[
  {
    "metric_name": "ttfb", // Time to First Byte in seconds
    "processor": "SmallestTTSService#0",
    "mean": 0.3538844585418701,
    "std": 0.026930570602416992,
    "values": [0.3808150291442871, 0.3269538879394531] // values for each text input
  },
  {
    "metric_name": "processing_time", // Time taken by the service to respond in seconds
    "processor": "SmallestTTSService#0",
    "mean": 0.00022804737091064453,
    "std": 2.7060508728027344e-5,
    "values": [0.0002009868621826172, 0.0002551078796386719] // values for each text input
  }
]
```

You can checkout [`src/tts/examples/sample_output`](src/tts/examples/sample_output) to see a sample output of the evaluation script.

For more details, run `uv run python eval.py -h`.

```bash
usage: eval.py [-h] [-p {smallest,cartesia,openai,groq,google,elevenlabs,elevenlabs-http,sarvam,sarvam-http}]
               [-l {english,hindi}] -i INPUT [-o OUTPUT_DIR] [-d]

options:
  -h, --help            show this help message and exit
  -p {smallest,cartesia,openai,groq,google,elevenlabs,elevenlabs-http,sarvam,sarvam-http}, --provider {smallest,cartesia,openai,groq,google,elevenlabs,elevenlabs-http,sarvam,sarvam-http}
                        TTS provider to use for evaluation
  -l {english,hindi}, --language {english,hindi}
                        Language of the audio files
  -i INPUT, --input INPUT
                        Path to the input CSV file containing the texts to synthesize
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Path to the output directory to save the results
  -d, --debug           Run the evaluation on the first 5 audio files
```

To benchmark several TTS runs, generate the combined workbook and chart with:

```bash
cd src/tts
uv run python leaderboard.py -o /path/to/output -s ./leaderboards
```

`src/tts/leaderboard.py` mirrors the STT workflow, emitting `tts_leaderboard.xlsx` plus `all_metrics_by_run.png` so you can spot which provider balances latency and judge scores best.

You can checkout [`src/tts/examples/leaderboard`](src/tts/examples/leaderboard) to see a sample output of the leaderboard script.

![Leaderboard example](src/tts/examples/leaderboard/all_metrics_by_run.png)

## LLM tests

To evaluate LLM behavior through test cases, prepare a JSON configuration file with test cases and evaluation criteria.

The test configuration file should have the following structure:

```json
{
  "params": {
    "model": "gpt-4.1" // the name of the openai model
  },
  "system_prompt": "You are a helpful assistant...", // the system prompt for the LLM. Can include a placeholder for `{language}` that will be replaced with possible values from `variables`
  "tools": [
    // array of tools that the LLM can call. Each tool should have:
    {
      "type": "client", // the type of the tool. Currently only "client" is supported
      "name": "plan_next_question", // the name of the tool
      "description": "Tool description...", // the description of the tool
      "parameters": [
        // array of parameters that the tool accepts
        {
          "id": "next_unanswered_question_index", // the id of the parameter
          "type": "integer", // the type of the parameter
          "description": "Parameter description...", // the description of the parameter
          "required": true // whether the parameter is required
        }
      ]
    }
  ],
  "variables": {
    "language": ["hindi", "english"] // array of language values to iterate over. The system prompt will be formatted with these values
  },
  "test_cases": [
    // array of test cases, each containing:
    {
      "history": [
        // the conversation history uptil the point of the test
        {
          "role": "assistant",
          "content": "Hello! What is your name?"
        },
        {
          "role": "user",
          "content": "Aman Dalmia"
        }
      ],
      "evaluation": {
        // the evaluation criteria for the test
        "type": "tool_call", // the type of evaluation. Either "tool_call" or "response". "tool_call" means that the LLM's tool calls will be evaluated against the expected tool calls.
        "tool_calls": [
          // array of tool calls that the LLM is expected to make. Each tool call should have:
          {
            "tool": "plan_next_question", // the name of the tool
            "arguments": {
              // the arguments that the tool accepts
              "next_unanswered_question_index": 2, // the value of the parameter
              "questions_answered": [1] // the value of the parameter
            }
          }
        ]
      }
    },
    {
      "history": [
        {
          "role": "assistant",
          "content": "Hello! Let's fill out your ANC form. What is your name?"
        },
        {
          "role": "user",
          "content": "Aman Dalmia"
        },
        {
          "role": "assistant",
          "content": "Thanks, what is your phone number?"
        },
        {
          "role": "user",
          "content": "Can I skip this question?"
        }
      ],
      "evaluation": {
        "type": "response", // `response` means that the LLM response will be evaluated using an LLM judge with custom criteria
        "criteria": "The assistant should allow the user to skip giving their phone number if asked to skip, and not press them further to answer or call the plan_next_question tool for that question.", // the criteria for evaluating the agent response
        "positive_examples": [], // array of positive examples that the LLM is expected to respond with
        "negative_examples": [] // array of negative examples that the LLM is expected to respond with
      }
    }
  ]
}
```

Copy `src/llm/.env.example` to `src/llm/.env` and fill in the API keys for the providers you want to evaluate.

```bash
cd src/llm
uv run python eval.py --config /path/to/test_config.json -o /path/to/output_dir -m <model_name>
```

You can use the sample test configuration provided in [`src/llm/examples/tests/config.json`](src/llm/examples/tests/config.json) to test the evaluation script.

```bash
uv run python run_tests.py --config examples/tests/config.json -o ./out -m gpt-4.1
```

The script will run each test case for each variable value (e.g., for each language) and output:

- Pass/fail status for each test case
- Summary statistics showing total passed/failed tests
- Detailed output for failed test cases including the test case, actual output, and evaluation metrics

The output of the script will be saved in the output directory.

```bash
/path/to/output/<test_config_name>/<model_name>
├── results.json
├── logs
```

You can checkout [`src/llm/examples/tests/sample_output`](src/llm/examples/tests/sample_output) to see a sample output of the evaluation script.

Sample output 1:

```
✅ Test case 1 passed
----------------------------------------
✅ Test case 2 passed
----------------------------------------
🎉 All tests passed!
```

Sample output 2:

```
❌ Test case 1 failed
----------------------------------------
✅ Test case 2 passed
----------------------------------------
✅ Total Passed: 1/2 (50.0%)
❌ Total Failed: 1/2 (50.0%)
Failed test cases:
========================================
History:


assistant: Namaste, I’m Nurse Anitha. To start, what name should I write for you?
user: My name is Geetha Shankar. I am 24 years old and live in Hosahalli village, Holenarasipura taluk, Hassan district


Expected output:
{'type': 'tool_call', 'tool_calls': [{'tool': 'plan_next_question', 'arguments': {'next_unanswered_question_index': 3, 'questions_answered': [1, 2, 4]}}]}
----------------------------------------
Output:
{'response': '', 'tool_calls': [{'tool': 'plan_next_question', 'arguments': {'next_unanswered_question_index': 3, 'questions_answered': [1, 2]}}, {'tool': 'plan_next_question', 'arguments': {'next_unanswered_question_index': 4, 'questions_answered': [1, 2]}}]}
----------------------------------------
Metrics:
{'passed': False}
----------------------------------------
========================================
```

Once you have results for multiple models or scenarios, compile a leaderboard CSV and comparison chart:

```bash
cd src/llm
uv run python tests_leaderboard.py -o /path/to/output -s ./leaderboard
```

`src/llm/tests_leaderboard.py` walks every `<scenario>/<model>` folder under the output root, computes pass percentages, and saves `llm_leaderboard.csv` plus `llm_leaderboard.png` to the chosen save directory for quick reporting.

You can checkout [`src/llm/examples/leaderboard`](src/llm/examples/leaderboard) to see a sample output of the leaderboard script.

![Leaderboard example](src/llm/examples/tests/leaderboard/llm_leaderboard.png)

## LLM simulations

Run fully automated, text-only conversations between two LLMs—the “agent” plus multiple users having specific personas to mimic specific scenarios given by you. Prepare a JSON configuration file as shown below:

```json
{
  "agent_system_prompt": "instructions for the bot",
  "language": "english", // language of the conversation
  "tools": [...], // tool schemas available to the agent LLM
  "personas": [...], // different personas for users in the simulation
  "scenarios": [...], // different simulation scenarios to be run with each user persona
  "evaluation_criteria": [
    // array of criteria to evaluate each simulation against
    {
      "name": "question_completeness", // unique name for the criterion
      "description": "Whether all the questions in the form were covered in the conversation" // description used by LLM judge to evaluate
    },
    {
      "name": "assistant_behavior",
      "description": "Whether the assistant asks one concise question per turn; is empathetic and does not repeat the user's answer back to them."
    }
  ]
}
```

Each simulation pairs every persona with every scenario. The runner fans out into separate folders named `simulation_persona_<n>_scenario_<m>` and saves transcripts, evaluation results, plus logs for inspection. Use the sample persona/scenario set from the example config or replace them with your own scripted behaviors (e.g., "refuses to share phone number until probed twice"). Use these artifacts to verify that the agent follows your instructions (e.g., responds appropriately, calls the right tools at the right times, and terminates the call correctly), then iterate on the prompts/config as needed.

After each simulation, an LLM judge evaluates the transcript against all `evaluation_criteria` defined in the config. Each criterion produces a match (true/false) and reasoning. The results are aggregated across all simulations at the end of the run.

Prepare the `.env` file with your OpenAI key (`cp src/llm/.env.example src/llm/.env`), then launch simulations:

```bash
cd src/llm
uv run python run_simulation.py \
  --config /path/to/config.json \
  --output-dir /path/to/output \
  --model <model_name>
```

You can use the sample configuration provided in [`src/llm/examples/simulation/config.json`](src/llm/examples/simulation/config.json) to test the simulation.

```bash
cd src/llm
uv run python run_simulation.py \
  --config examples/simulation/config.json \
  --output-dir ./out \
  --model gpt-4.1
```

The output of the simulation will be saved in the output directory:

```bash
/path/to/output
├── simulation_persona_1_scenario_1
│   ├── transcript.json          # full conversation transcript
│   ├── evaluation_results.csv   # per-criterion evaluation results
│   └── logs                     # timestamped pipeline logs
├── simulation_persona_1_scenario_2
│   └── ...
├── results.csv                  # aggregated results for all simulations
└── metrics.json                 # summary statistics for each criterion
```

`evaluation_results.csv` contains the evaluation results for each criterion in a simulation:

```csv
name,match,reasoning
question_completeness,True,"The assistant asked for the user's full name, address, and telephone number..."
assistant_behavior,True,"The assistant asked one concise question per turn..."
```

`results.csv` aggregates the match scores across all simulations:

```csv
name,question_completeness,assistant_behavior
simulation_persona_1_scenario_1,1.0,1.0
simulation_persona_1_scenario_2,1.0,1.0
simulation_persona_1_scenario_3,1.0,1.0
```

`metrics.json` provides summary statistics (mean, std, values) for each evaluation criterion:

```json
{
  "question_completeness": {
    "mean": 1.0,
    "std": 0,
    "values": [1.0, 1.0, 1.0]
  },
  "assistant_behavior": {
    "mean": 1.0,
    "std": 0.0,
    "values": [1.0, 1.0, 1.0]
  }
}
```

You can checkout [`src/llm/examples/simulation/sample_output`](src/llm/examples/simulation/sample_output) to see a sample output of the simulation.

## Voice agent simulations

To run agent simulations with all the three components - STT, LLM, and TTS - prepare a JSON configuration file as shown below.

```json
{
  "agent_system_prompt": "You are a helpful nurse speaking to a pregnant mother who has come for an Antenatal visit (ANC visit).\n\nYou are helping her with filling the ANC visit form which has the following questions:\n\n1: Name of patient (string)\n2: Address (string)\n3: Telephone (number)\n\nYou always speak in english.\n\nYour goal is to get the answer for all these questions from the user. Ask the questions sequentially, one at a time. Make sure to always speak out the next question for the user. They don't know what the next question will be. It is your responsibility to ask them the next question.\n\nDo not repeat the user's answer back to them.\n\nExcept for the questions where a type has been explicitly given beside it, the rest of the questions are boolean questions (yes/no).\n\nIf the user gives an answer that is not valid for a question, then, don't call the `plan_next_question` tool at all.\n\nOnly if the user gives a valid answer to a question, then, call the `plan_next_question` tool.\n\nOnce all the questions have been answered, end the call by calling the `end_call` tool immediately without asking or saying anything else to the user.\n\n# Important instructions\nFor each field, you must make sure that you get valid and complete answers from the user.\n\nName: the full name including both the first name and last name (users might have initials as part of their name - don't ask them to expand it)\nAddress: the full address of the user's specific place of residence\nTelephone: must be a valid 10-digit phone number (don't bug the user to say the number in 10 digits or without spaces or without hyphens or to repeat it as long you can infer the number from your message, irrespective of whatever format it may have been given in - asking for repeating information that is already clear is not a good user experience)\n\nUntil you get a valid and complete response for a specific question from the user, keep probing for more details until you have extracted all the details required to meet the criteria of completeness for that question.\n\nFor the boolean questions, just look for whether the user has that symptom or not. If the user is not able to give a definitive answer to any the boolean questions, try probing once to help them arrive at a definitive answer. If they are still not able to give a definitive true/false answer, mark that response as null.\n\n# Skipping already answered questions\nIt is possible that even though you have asked a particular question, the user's response may end up answering some of the future questions you were going to ask. If the user's response already definitively answers those questions, then, don't repeat those questions again and skip them to move to the next unanswered question.\n\nAfter every valid and complete response given by the user to a question, call the `plan_next_question` tool with the indices of the questions in the list of questions above that the user's response has already answered and the index of the next unanswered question that you are going to ask.\n\nIf the user's response only answers the exact question you asked, then, call `plan_next_question` with `questions_answered` as `[<index_of_that_question>]` and `next_unanswered_question_index` as the index of the next unanswered question based on the conversation history so far.\n\nIf the user's response answers more questions beyond the question you asked, then, call `plan_next_question` with `questions_answered` as `[<index_of_the_first_question_answered>, <index_of_the_second_question_answered>, ....]` and `next_unanswered_question_index` as the index of the next unanswered question based on the conversation history so far.\n\nIf the user's response answers some other question from the form but not the question you asked, then, call `plan_next_question` with `questions_answered` as `[<index_of_the_question_answered>]` and `next_unanswered_question_index` as the index of the question you had originally asked but the user did not answer.\n\nIf the user's response is completely irrelevant to any of the questions in the form, don't call this tool and steer the user back to the conversation.\n\nUse the same index values as mentioned in the list above: from 1-22. So, 1-indexed. Not, 0-indexed.\n\n# Ensuring completion of all questions\n- If the user insists on skipping any question when you asked it, make sure to ask it again later. Before ending the call, make sure to ask all the questions that you asked. Only when all the questions have been asked and answered, then, call the `end_call` tool.\n\n# Style\n- Keep the conversation clear, warm, friendly and empathetic. It should feel like a natural and realistic conversation between a nurse and a mother who is pregnant. Make sure each question that you ask is concise and to the point.\n- Speak in an empathetic, friendly tone like a nurse at a government hospital.\n- Always stay gender neutral and don't say anything that might indicate any gender or name for you or anthropomorphise you in any way.\n", // prompt for the voice agent
  "language": "english", // language of the conversation
  "tools": [
    {
      "type": "client",
      "name": "plan_next_question",
      "description": "Optional, only call this tool if the user gave a valid answer to a valid question that you asked; if yes, then which questions have been answered and which question needs to be asked next; if not, don't call this tool at all",
      "disable_interruptions": false,
      "force_pre_tool_speech": "auto",
      "assignments": [],
      "tool_call_sound": null,
      "tool_call_sound_behavior": "auto",
      "execution_mode": "immediate",
      "expects_response": false,
      "response_timeout_secs": 1,
      "parameters": [
        {
          "id": "next_unanswered_question_index",
          "type": "integer",
          "value_type": "llm_prompt",
          "description": "the index of the next unanswered question that should be asked to the user next. don't repeat questions that have been asked before already in the previous responses.",
          "dynamic_variable": "",
          "constant_value": "",
          "required": true
        },
        {
          "id": "questions_answered",
          "type": "array",
          "description": "Optional, the indices of the questions in the full list of questions that have been answered by the user's last response",
          "items": {
            "type": "integer",
            "description": "the index of each question that got answered by the current response of the user"
          },
          "required": true,
          "value_type": "llm_prompt"
        }
      ],
      "dynamic_variables": {
        "dynamic_variable_placeholders": {}
      }
    }
  ], // tools that the voice agent has access to
  "personas": [
    "You are a mother; name is Geeta Prasad, 39 years old lives at Flat 302, Sri Venkateshwara Nilaya, near Hanuman Temple, Indiranagar. Phone number is plus nine one, nine eight triple three, forty-seven twenty-nine zero; never had a stillbirth or a baby who died soon after birth, but has had one baby who died on the third day after delivery; never had three or more miscarriages in a row; in her last pregnancy she wasn’t admitted for high blood pressure or eclampsia; only carrying one baby as per the scan; blood group is O positive, so there are no Rh-related issues; did experience light vaginal spotting once during this pregnancy but otherwise no pelvic mass or complications; uncertain about her exact blood-pressure reading, but recalls that it was around 150/95 mmHg at booking; does not have diabetes, heart disease, kidney disease, or epilepsy; does have asthma and uses an inhaler daily; has never had tuberculosis or any other serious medical condition; longer drinks alcohol or uses substances, having stopped completely after learning she was pregnant; is very shy and reserved and uses short answers to questions"
  ], // different user personas to evaluate for the voice agent
  "scenarios": [
    "the mother answers all the questions and completes the form, don't ask any further questions after the form is filled",
    "the mother hesitates in directly answering some questions and wants to skip answering them at first but answers later on further probing",
    "the mother is in a rush and ends up answering multiple questions at once that are related to each other when being asked one question; the list of all questions - Name of patient, Address, Telephone",
    "mother ends up occasionally giving partial or invalid answers and corrects them later on in her immediate next message instead of answering the question that she was asked on that turn; first give an incomplete/invalid response and wait for the agent's next message before correcting your previous response"
  ] // the different scenarios to evaluate for each persona
}
```

The voice agent is defined in `src/agent/bot.py`. For now, the default configuration of the bot uses Deepgram for STT, OpenAI for LLM, and Google for TTS. You can configure the bot to use different providers for STT, LLM, and TTS by modifying the `STTConfig`, `LLMConfig`, and `TTSConfig` classes in `src/agent/bot.py`. We will soon add support for changing the configuration for each provider for each component to override the defaults.

Copy `src/agent/.env.example` to `src/agent/.env` and fill in the API keys for the providers you want to evaluate.

```bash
cd src/agent
uv run python eval.py  -c /path/to/config.json -o /path/to/output
```

You can use the sample configuration provided in [`src/agent/examples/sample_short.json`](src/agent/examples/sample_short.json) to test the voice agent simulation.

```bash
uv run python eval.py -c examples/sample_short.json -o ./out
```

The output of the evaluation script will be saved in the output directory.

```bash
/path/to/output/
├── simulation_persona_1_scenario_1
│   ├── audios
│   │    ├── 0_user.wav
│   │    ├── 1_bot.wav
│   │    ├── 1_user.wav
│   │    ...
│   ├── logs
│   ├── metrics.json
│   ├── stt_outputs.json
│   ├── tool_calls.json
│   ├── transcripts.json
├── simulation_persona_1_scenario_2
├── simulation_persona_1_scenario_3
└── ...
```

Each `simulation_persona_*_scenario_*` directory contains:

- `audios/`: alternating `*_user.wav` and `*_bot.wav` clips for every turn so you can audit microphone and synthesis quality at each step.
- `logs`: timestamped pipeline logs (component + severity) that make it easy to trace pipeline startup and per-turn processing.
- `metrics.json`: latency traces grouped by processor for both `ttft` (time to first token) and `processing_time`, reported separately for STT, LLM, and TTS providers.
- `stt_outputs.json`: The output of the STT step for each turn in the voice agent being evaluated.
- `tool_calls.json`: chronologically ordered tool calls made by the agent.
- `transcripts.json`: full conversation transcript - alternating `user`/`assistant` turns.

You can checkout [`src/agent/examples/simulation/sample_output`](src/agent/examples/simulation/sample_output) to see a sample output of the voice agent simulation.

## TODO

- For STT and TTS, save the pipecat logs to a file and don't show in the terminal - only show key milestones in the terminal.
- Support for configuring interrupts in the voice agent simulation
- Make visualizations using tools like INK on the agent flow
- Support for adding knowledge base to agent/llm simulations
- Reuse code blocks between different modules instead of duplicating code.
- Store transcripts and tool calls in the output directory when talking to an agent.
- Support for adding custom config for each provider for each component to override the defaults.

## Bugs

- The STT module doesn't run on all the audio files for some reason and currently, the number of audios it runs on varies for each run. The transcript captured is sometimes incomplete as well where only the last text chunk is captured.
