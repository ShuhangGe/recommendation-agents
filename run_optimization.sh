#!/bin/bash
# Script to run a full optimization cycle with configurable models

# Default values
RECOMMENDATION_MODEL="gemini-1.5-flash"
EVALUATION_MODEL="gemini-1.5-pro"
OPTIMIZER_MODEL="gemini-1.5-pro"
MAX_ITERATIONS=5

# Parse command-line options
while [[ $# -gt 0 ]]; do
  case $1 in
    --recommendation-model)
      RECOMMENDATION_MODEL="$2"
      shift 2
      ;;
    --evaluation-model)
      EVALUATION_MODEL="$2"
      shift 2
      ;;
    --optimizer-model)
      OPTIMIZER_MODEL="$2"
      shift 2
      ;;
    --iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --recommendation-model MODEL   Model for recommendations (default: gemini-1.5-flash)"
      echo "  --evaluation-model MODEL       Model for evaluations (default: gemini-1.5-pro)"
      echo "  --optimizer-model MODEL        Model for optimizations (default: gemini-1.5-pro)"
      echo "  --iterations NUM               Maximum iterations (default: 5)"
      echo "  --help                         Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Starting optimization with the following configuration:"
echo "- Recommendation model: $RECOMMENDATION_MODEL"
echo "- Evaluation model: $EVALUATION_MODEL"
echo "- Optimizer model: $OPTIMIZER_MODEL"
echo "- Max iterations: $MAX_ITERATIONS"

# Run the main script with specified models
python main.py \
  --recommendation-model "$RECOMMENDATION_MODEL" \
  --evaluation-model "$EVALUATION_MODEL" \
  --optimizer-model "$OPTIMIZER_MODEL" \
  --max-iterations "$MAX_ITERATIONS"

echo "Optimization complete. Check the results directory for output." 