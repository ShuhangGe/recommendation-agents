#!/bin/bash
# Script to generate stories using OpenAI models

# Default values
MODEL="gpt-3.5-turbo"
COUNT=100
REGEN=false

# Parse command-line options
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --count)
      COUNT="$2"
      shift 2
      ;;
    --regenerate)
      REGEN=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --model MODEL     Model to use for generation (default: gpt-3.5-turbo)"
      echo "  --count COUNT     Number of stories to generate (default: 100)"
      echo "  --regenerate      Force regeneration of stories"
      echo "  --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Generating stories using model: $MODEL"

# Generate regeneration flag if needed
REGEN_FLAG=""
if $REGEN; then
  REGEN_FLAG="--regenerate-data"
fi

# Run the main script with story generation only
python main.py --story-model "$MODEL" --story-count "$COUNT" $REGEN_FLAG --max-iterations 0

echo "Story generation complete. Check the data directory for results." 