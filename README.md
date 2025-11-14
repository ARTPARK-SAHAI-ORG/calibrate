# AgentLoop

Evaluation suite for voice agents.

AgentLoop gives you the flexibility to evaluate different components of the voice agent pipeline in isolation while also running simulations and tests on the agent as a whole as well.

AgentLoop is built on top of [pipecat](https://github.com/pipecat-ai/pipecat), a framework for building voice agents.

## Setup

We use `uv` to manage the dependencies. Make sure you have it installed on your system.

Create the virtual environment and install the dependencies.

```bash
uv sync --frozen
```

## Quickstart

### Speech To Text (STT)

To evaluate different STT providers, first make sure to organize the input data in the following structure:

```bash
├── /path/to/data
│   └── gt.json
│   └── audio
│       └── audio_1.wav
│       └── audio_2.wav
│   └── audio_pcm16
│       └── audio_1.wav
│       └── audio_2.wav
```

`gt.json` should have the following format:

```json
{
  "audio_1": "<transcription_1>",
  "audio_2": "<transcription_2>"
}
```

Store the audios in both `wav` and `pcm16` formats as different providers support different formats. The evaluation script will automatically select the right format for the provider you choose to evaluate.

Copy `src/stt/.env.example` to `src/stt/.env` and fill in the API keys for the providers you want to evaluate.

```bash
cd src/stt
uv run python eval.py --input-dir /path/to/data -o /path/to/output -d -p <provider> -l <language>
```

You can use the sample inputs provided in [`src/stt/samples`](src/stt/samples) to test the evaluation script.

```bash
cd src/stt
uv run python eval.py --input-dir samples -o ./out -d -p sarvam -l english
```

The output of the evaluation script will be saved in the output directory.

```bash
/path/to/output
├── sarvam_english
│   ├── logs
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

### Text To Speech (TTS)

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

You can use the sample inputs provided in [`src/tts/examples`](src/tts/examples) to test the evaluation script.

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

### LLM simulations

Coming Soon

### LLM tests

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
uv run python run_tests.py --config examples/tests.json -o ./out
```

You can use the sample test configuration provided in [`src/llm/examples/tests.json`](src/llm/examples/tests.json) to test the evaluation script.

The script will run each test case for each variable value (e.g., for each language) and output:

- Pass/fail status for each test case
- Summary statistics showing total passed/failed tests
- Detailed output for failed test cases including the test case, actual output, and evaluation metrics

For more details, run `uv run python run_tests.py -h`.

```bash
usage: run_tests.py [-h] [--config CONFIG] [-o OUTPUT_DIR] [-d]

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the JSON configuration file for the tests (default: examples/tests.json)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Path to the output directory to save the results (default: ./out)
  -d, --debug           Run the evaluation on the first 5 audio files
```

Sample output:

```
----------------------------------------------------------------------------------------------------
✅ Total Passed: 3/4 (75.0%)
❌ Total Failed: 1/4 (25.0%)
Failed test cases:
====================================================================================================
Test case:
{'history': [{'role': 'assistant', 'content': "Hello! Let's fill out your ANC form. What is your name?"}, {'role': 'user', 'content': 'Aman Dalmia'}, {'role': 'assistant', 'content': 'Thanks, what is your phone number?'}, {'role': 'user', 'content': 'Can I skip this question?'}], 'evaluation': {'type': 'response', 'criteria': 'the assistant must let the user skip the plan_next_question', 'positive_examples': [], 'negative_examples': []}}
----------------------------------------------------------------------------------------------------
Output:
{'response': 'Aapka phone number lena zaroori hai taki hum aapko zarurat padne par contact kar sakein. Kripya apna 10 digit phone number batayein.', 'tool_calls': []}
----------------------------------------------------------------------------------------------------
Metrics:
{'passed': False, 'reasoning': 'The response does not adhere to the evaluation criteria. The assistant insists that the user provide their phone number, stating it is necessary for contact, and does not allow the user to skip the question. The criteria clearly state that the assistant must let the user skip the question and proceed to the next one, which is not done here.'}
----------------------------------------------------------------------------------------------------
=====================
```

### Agent simulations

Coming Soon

### Talk to the agent

Coming Soon

## TODO

- Add evaluation metrics for quality of TTS
- Add documentation for LLM tests and simulations
- Add script for voice agent simulation
- Add script for talking to the agent
- Support for adding custom config for each provider for each component to override the defaults.
- Make visualizations using tools like INK on the agent flow
