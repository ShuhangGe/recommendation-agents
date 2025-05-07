#!/bin/bash
# Script to generate data for the Sekai Recommendation Agent system
#
# This script generates story and user profile data for the Sekai Recommendation Agent.
# The original seed data from the Sekai Take-Home Challenge (5 stories and 3 user profiles)
# are always preserved as "ground truth" examples in the generated data.

# Default values
STORY_MODEL="gpt-4o"
USER_MODEL="gpt-4o"
STORY_COUNT=100
USER_COUNT=5
STORIES_PER_USER=20
REGENERATE=false

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --story-model MODEL     Model to use for story generation (default: ${STORY_MODEL})"
    echo "  --user-model MODEL      Model to use for user profile generation (default: ${USER_MODEL})"
    echo "  --story-count COUNT     Total number of stories to generate, including ground truth examples (default: ${STORY_COUNT})"
    echo "  --user-count COUNT      TOTAL number of users to have, including the 3 ground truth users (default: ${USER_COUNT})"
    echo "  --stories-per-user NUM  Number of stories to generate specifically for EACH user's interests,"
    echo "                          ensuring EQUAL representation across different preferences (default: ${STORIES_PER_USER})"  
    echo "  --regenerate            Force regeneration of data even if it exists"
    echo "  --list-models           List available models"
    echo "  --help                  Show this help message"
    echo ""
    echo "Note: The original 5 stories and 3 user profiles from the Sekai Take-Home Challenge"
    echo "      are ALWAYS preserved as ground truth examples in the generated data."
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --story-model)
            STORY_MODEL="$2"
            shift
            shift
            ;;
        --user-model)
            USER_MODEL="$2"
            shift
            shift
            ;;
        --story-count)
            STORY_COUNT="$2"
            shift
            shift
            ;;
        --user-count)
            USER_COUNT="$2"
            shift
            shift
            ;;
        --stories-per-user)
            STORIES_PER_USER="$2"
            shift
            shift
            ;;  
        --regenerate)
            REGENERATE=true
            shift
            ;;
        --list-models)
            python -c "import utils.model_utils; print('Available models:')\nprint('\n'.join(utils.model_utils.list_available_models()))"
            exit 0
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            usage
            exit 1
            ;;
    esac
done

# Calculate how many additional users will be generated
ADDITIONAL_USERS=$((USER_COUNT - 3))
if [ "$ADDITIONAL_USERS" -lt 0 ]; then
    ADDITIONAL_USERS=0
fi

# Print configuration
echo "🛠️  Data Generation Configuration"
echo "--------------------------------"
echo "📚 Story model:       $STORY_MODEL"
echo "👤 User model:        $USER_MODEL"
echo "📊 Story count:       $STORY_COUNT (including 5 ground truth examples)"
echo "👥 User count:        $USER_COUNT total ($ADDITIONAL_USERS additional + 3 ground truth examples)"
echo "🎯 Stories per user:  $STORIES_PER_USER (ensuring equal representation of interests)"
echo "🔄 Regenerate data:   $REGENERATE"
echo ""

# Build the command - handle the regenerate flag correctly
REGENERATE_ARG=""
if [ "$REGENERATE" = true ]; then
    REGENERATE_ARG="--regenerate"
fi

# Run the data generator
echo "🚀 Starting data generation process..."
python generate_data.py \
    --story-model "$STORY_MODEL" \
    --user-model "$USER_MODEL" \
    --story-count $STORY_COUNT \
    --user-count $USER_COUNT \
    --stories-per-user $STORIES_PER_USER \
    $REGENERATE_ARG

echo -e "\n✅ Data generation complete!"
echo -e "   The ground truth examples from the Sekai Take-Home Challenge have been preserved."
echo -e "   Generated $STORIES_PER_USER stories specifically tailored to each user's interests." 