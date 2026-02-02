#!/bin/bash

# Benchmark script to run STT evaluation for all providers in parallel

set -e

# Required arguments
INPUT_DIR=""
OUTPUT_DIR=""
LANGUAGE=""
BASE_PORT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -l|--language)
            LANGUAGE="$2"
            shift 2
            ;;
        -p|--base-port)
            BASE_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -i <input-dir> -o <output-dir> -l <language> -p <base-port>"
            echo ""
            echo "Options:"
            echo "  -i, --input-dir   Path to input directory containing audio files and stt.csv (required)"
            echo "  -o, --output-dir  Path to output directory (required)"
            echo "  -l, --language    Language of the audio files: english, hindi (required)"
            echo "  -p, --base-port   Starting port number for parallel providers (required)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_DIR" ]; then
    echo "Error: Input directory is required"
    echo "Usage: $0 -i <input-dir> -o <output-dir> -l <language> -p <base-port>"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory is required"
    echo "Usage: $0 -i <input-dir> -o <output-dir> -l <language> -p <base-port>"
    exit 1
fi

if [ -z "$LANGUAGE" ]; then
    echo "Error: Language is required"
    echo "Usage: $0 -i <input-dir> -o <output-dir> -l <language> -p <base-port>"
    exit 1
fi

if [ -z "$BASE_PORT" ]; then
    echo "Error: Base port is required"
    echo "Usage: $0 -i <input-dir> -o <output-dir> -l <language> -p <base-port>"
    exit 1
fi

# List of providers to benchmark
PROVIDERS=(
    "deepgram"
    "openai"
    "cartesia"
    "elevenlabs"
    # "smallest"
    "groq"
    "google"
    "sarvam"
)

echo "Starting STT benchmark..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Language: $LANGUAGE"
echo "Base port: $BASE_PORT"
echo "Providers: ${PROVIDERS[*]}"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Array to store background process PIDs
PIDS=()

# Run each provider in parallel on different ports
for i in "${!PROVIDERS[@]}"; do
    PROVIDER="${PROVIDERS[$i]}"
    PORT=$((BASE_PORT + i))
    
    echo "Starting $PROVIDER on port $PORT..."
    
    uv run python eval.py \
        --provider "$PROVIDER" \
        --language "$LANGUAGE" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --port "$PORT" \
        > "$OUTPUT_DIR/${PROVIDER}_${LANGUAGE}.log" 2>&1 &
    
    PIDS+=($!)
done

echo ""
echo "All providers started. Waiting for completion..."
echo ""

# Wait for all background processes and track failures
FAILED=()
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    PROVIDER="${PROVIDERS[$i]}"
    
    if wait "$PID"; then
        echo "✓ $PROVIDER completed successfully"
    else
        echo "✗ $PROVIDER failed"
        FAILED+=("$PROVIDER")
    fi
done

echo ""
echo "Benchmark complete!"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed providers: ${FAILED[*]}"
    exit 1
else
    echo "All providers completed successfully"
fi

