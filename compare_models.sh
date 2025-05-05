#!/bin/bash
# Script to compare different models for a specific task

# Default values
TASK="recommendation"
USER_ID="user1"
MODELS=("gpt-3.5-turbo" "gpt-4" "gemini-1.5-flash" "gemini-1.5-pro")
JSON_OUTPUT=false

# Parse command-line options
while [[ $# -gt 0 ]]; do
  case $1 in
    --task)
      TASK="$2"
      shift 2
      ;;
    --user-id)
      USER_ID="$2"
      shift 2
      ;;
    --models)
      IFS=',' read -r -a MODELS <<< "$2"
      shift 2
      ;;
    --json)
      JSON_OUTPUT=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --task TASK       Task to test (recommendation, evaluation, tags, optimization)"
      echo "  --user-id USER_ID User ID to test with (default: user1)"
      echo "  --models MODELS   Comma-separated list of models to compare (default: gpt-3.5-turbo,gpt-4,gemini-1.5-flash,gemini-1.5-pro)"
      echo "  --json            Output results in JSON format"
      echo "  --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Comparing models for task: $TASK"
echo "User ID: $USER_ID"
echo "Models to compare: ${MODELS[*]}"

# Check if we should use JSON output
JSON_FLAG=""
if $JSON_OUTPUT; then
  JSON_FLAG="--json-output"
fi

# Create results directory
RESULTS_DIR="model_comparison_results"
mkdir -p "$RESULTS_DIR"

# Run tests with each model
for model in "${MODELS[@]}"; do
  echo "------------------------------------------------------"
  echo "Testing model: $model"
  echo "------------------------------------------------------"
  
  # Run the test and capture output
  OUTPUT_FILE="$RESULTS_DIR/${TASK}_${model}_$(date +%Y%m%d%H%M%S).txt"
  python test_task.py --task "$TASK" --model "$model" --user-id "$USER_ID" $JSON_FLAG | tee "$OUTPUT_FILE"
  
  echo "Results saved to $OUTPUT_FILE"
  echo ""
done

echo "Comparison complete. All results saved in $RESULTS_DIR directory." 