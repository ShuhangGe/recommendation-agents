#!/bin/bash
# Script to run the agent-based adaptive recommendation system

# Default values
RECOMMENDATION_MODEL="gpt-4o-mini"
EVALUATION_MODEL="gpt-4o-mini"
USER_ID="user1"
VERBOSE=false
JSON_OUTPUT=false

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
    --user-id)
      USER_ID="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
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
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --recommendation-model MODEL   Model for recommendations (default: gpt-4o-mini)"
      echo "  --evaluation-model MODEL       Model for evaluations (default: gpt-4o-mini)"
      echo "  --user-id USER_ID              User ID to test (e.g., user1)"
      echo "  --profile TEXT                 User profile text (alternative to user-id)"
      echo "  --verbose                      Print detailed information about the process"
      echo "  --json                         Output results in JSON format"
      echo "  --help                         Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Starting agent-based recommendation system:"
echo "- Recommendation model: $RECOMMENDATION_MODEL"
echo "- Evaluation model: $EVALUATION_MODEL"
echo "- User: $USER_ID"

# Construct command
CMD="python run_agent_recommendation.py --recommendation-model \"$RECOMMENDATION_MODEL\" --evaluation-model \"$EVALUATION_MODEL\""

if [ ! -z "$USER_ID" ]; then
  CMD="$CMD --user-id \"$USER_ID\""
elif [ ! -z "$PROFILE" ]; then
  CMD="$CMD --profile \"$PROFILE\""
fi

if $VERBOSE; then
  CMD="$CMD --verbose"
fi

if $JSON_OUTPUT; then
  CMD="$CMD --json-output"
fi

# Run the command
echo "Executing: $CMD"
eval $CMD

echo "Agent-based recommendation process complete." 