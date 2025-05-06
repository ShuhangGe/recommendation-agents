#!/bin/bash
# Script to run an extended optimization cycle with enhanced parameters
# to overcome plateauing

# Default values
RECOMMENDATION_MODEL="gpt-4o-mini"
EVALUATION_MODEL="gpt-4o-mini"
OPTIMIZER_MODEL="gpt-4o" # Use the more powerful model for optimization
MAX_ITERATIONS=30
SCORE_THRESHOLD=0.9
IMPROVEMENT_THRESHOLD=0.004

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
    --threshold)
      SCORE_THRESHOLD="$2"
      shift 2
      ;;
    --improvement)
      IMPROVEMENT_THRESHOLD="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --recommendation-model MODEL   Model for recommendations (default: gpt-4o-mini)"
      echo "  --evaluation-model MODEL       Model for evaluations (default: gpt-4o-mini)"
      echo "  --optimizer-model MODEL        Model for optimizations (default: gpt-4o)"
      echo "  --iterations NUM               Maximum iterations (default: 30)"
      echo "  --threshold SCORE              Score threshold (default: 0.9)"
      echo "  --improvement VAL              Improvement threshold (default: 0.004)"
      echo "  --help                         Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Starting extended optimization with the following configuration:"
echo "- Recommendation model: $RECOMMENDATION_MODEL"
echo "- Evaluation model: $EVALUATION_MODEL"
echo "- Optimizer model: $OPTIMIZER_MODEL"
echo "- Max iterations: $MAX_ITERATIONS"
echo "- Score threshold: $SCORE_THRESHOLD"
echo "- Improvement threshold: $IMPROVEMENT_THRESHOLD"
echo ""
echo "This extended optimization uses:"
echo "1. Advanced plateau detection"
echo "2. Dynamic temperature settings"
echo "3. Enhanced prompt guidance"
echo "4. More iterations"
echo ""

# Set environment variables to override config settings
export IMPROVEMENT_THRESHOLD=$IMPROVEMENT_THRESHOLD

# Run the main script with specified models
python main.py \
  --recommendation-model "$RECOMMENDATION_MODEL" \
  --evaluation-model "$EVALUATION_MODEL" \
  --optimizer-model "$OPTIMIZER_MODEL" \
  --max-iterations "$MAX_ITERATIONS" \
  --score-threshold "$SCORE_THRESHOLD"

echo "Extended optimization complete. Check the results directory for output." 