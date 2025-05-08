#!/bin/bash
# Script to run the agent-based adaptive recommendation system

# Default values
RECOMMENDATION_MODEL="gpt-4o-mini"

EVALUATION_MODEL="gpt-4o"
OPTIMIZER_MODEL="gpt-4.1-2025-04-14"
VERBOSE=false
JSON_OUTPUT=false
ENABLE_OPTIMIZATION=false
MAX_ITERATIONS=20
SCORE_THRESHOLD=0.90
IMPROVEMENT_THRESHOLD=0.002

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
    --verbose)
      VERBOSE=true
      shift
      ;;
    --json)
      JSON_OUTPUT=true
      shift
      ;;
    --enable-optimization)
      ENABLE_OPTIMIZATION=true
      shift
      ;;
    --max-iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --score-threshold)
      SCORE_THRESHOLD="$2"
      shift 2
      ;;
    --improvement-threshold)
      IMPROVEMENT_THRESHOLD="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --recommendation-model MODEL   Model for recommendations (default: gpt-4o)"
      echo "  --evaluation-model MODEL       Model for evaluations (default: gpt-4o-mini)"
      echo "  --optimizer-model MODEL        Model for prompt optimization (default: gpt-4o)"
      echo "  --verbose                      Print additional information in the terminal"
      echo "  --json                         Output results in JSON format"
      echo "  --enable-optimization          Enable the prompt optimization loop"
      echo "  --max-iterations N             Maximum number of optimization iterations (default: 50)"
      echo "  --score-threshold N            Score threshold to stop optimization (default: 0.95)"
      echo "  --improvement-threshold N      Minimum improvement threshold - when below this, system reverts to best prompt instead of stopping (default: 0.001)"
      echo "  --help                         Show this help message"
      echo ""
      echo "Note: Terminal output will be concise and readable, while detailed logs are"
      echo "      automatically saved to a timestamped folder in the results directory."
      echo ""
      echo "Important: This script requires data to be generated first using ./generate_data.sh"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if data files exist
STORY_FILE="data/story_data.json"
USER_FILE="data/user_profiles.json"

if [ ! -f "$STORY_FILE" ] || [ ! -f "$USER_FILE" ]; then
  echo -e "\n❌ Error: Required data files not found."
  echo -e "Please generate data first using:"
  echo -e "  ./generate_data.sh"
  echo -e "\nExpected files:"
  echo -e "  $STORY_FILE"
  echo -e "  $USER_FILE"
  exit 1
fi

echo -e "\n🚀 Starting agent-based recommendation system:"
echo "- Recommendation model: $RECOMMENDATION_MODEL"
echo "- Evaluation model: $EVALUATION_MODEL"
echo "- Optimizer model: $OPTIMIZER_MODEL"

if $ENABLE_OPTIMIZATION; then
  echo "- Optimization enabled:"
  echo "  - Max iterations: $MAX_ITERATIONS"
  echo "  - Score threshold: $SCORE_THRESHOLD"
  echo "  - Improvement threshold: $IMPROVEMENT_THRESHOLD"
fi

echo -e "\n📝 Terminal will show concise output, detailed logs will be saved to results folder."
echo -e "----------------------------------------\n"

# Construct command
CMD="python main.py --recommendation-model \"$RECOMMENDATION_MODEL\" --evaluation-model \"$EVALUATION_MODEL\" --optimizer-model \"$OPTIMIZER_MODEL\""

if $VERBOSE; then
  CMD="$CMD --verbose"
fi

if $JSON_OUTPUT; then
  CMD="$CMD --json-output"
fi

if $ENABLE_OPTIMIZATION; then
  CMD="$CMD --enable-optimization --max-iterations $MAX_ITERATIONS --score-threshold $SCORE_THRESHOLD --improvement-threshold $IMPROVEMENT_THRESHOLD"
fi

# Run the command
eval $CMD

echo -e "\n✅ Agent-based recommendation process complete."
echo -e "   Check the timestamped results folder for detailed logs and outputs." 