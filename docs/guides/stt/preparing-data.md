# Preparing STT Data

Set up your test data for STT evaluation.

## Directory Structure

Organize your data as follows:

```
data/
├── stt.csv
├── audios/
│   ├── wav/
│   │   ├── audio_1.wav
│   │   └── audio_2.wav
│   └── pcm16/
│       ├── audio_1.wav
│       └── audio_2.wav
```

## CSV Format

`stt.csv` should have two columns:

```csv
id,text
audio_1,"Hi"
audio_2,"Madam, my name is Geeta Shankar"
```

- **id**: Matches the audio filename (without extension)
- **text**: Ground truth transcription

## Audio Formats

Provide audio in both formats:
- **wav**: Standard WAV format
- **pcm16**: PCM16 format (some providers require this)

Pense will automatically select the correct format for each provider.

## Next Steps

[Run your evaluation](running-eval.md).
