# AgentLoop

Evaluation suite for voice agents.

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
uv run python eval.py --input-dir /path/to/data -o /path/to/output -d -p sarvam -l hindi
```

The output of the evaluation script will be saved in the output directory.

```bash
/path/to/output
├── sarvam_hindi
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

Coming Soon

### Agent Simulations

Coming Soon

### Agent Tests

Coming Soon

```

```
