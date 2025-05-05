# Sekai Recommendation Agent System

A multi-agent recommendation system for interactive stories that leverages both Google's Gemini and OpenAI models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Scripts Explained](#scripts-explained)
- [Available Models](#available-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

The Sekai Recommendation Agent System is an advanced recommendation engine designed to suggest relevant interactive stories to users based on their preferences. The system employs multiple agent types working in conjunction to continuously optimize recommendations through an autonomous feedback loop.

Key capabilities:
- Personalized story recommendations based on user preferences
- Multi-model support (OpenAI and Google Gemini)
- Automatic prompt optimization
- Comprehensive evaluation metrics
- Extensible architecture for new models and features

## ✨ Features

- **Model Flexibility**: Seamlessly switch between different AI models (OpenAI GPT models and Google Gemini models)
- **Task-Specific Model Selection**: Specify different models for different tasks (recommendations, evaluations, optimizations)
- **Prompt Optimization**: Autonomous improvement of recommendation prompts based on performance
- **Caching System**: Efficient caching for embeddings and successful prompts
- **Comprehensive Evaluation**: Multiple metrics for measuring recommendation quality
- **User Profile Analysis**: Extract user preferences from freeform text profiles
- **Story Generation**: Generate synthetic story data for testing and training
- **Visualization**: Track optimization progress with result metrics

## 🏗️ System Architecture

The system consists of three primary agent types:

1. **Recommendation Agent**: Generates story recommendations for users based on their preference tags
   - Accepts any compatible model for inference
   - Uses a prompt-based approach for story ranking

2. **Evaluation Agent**: Assesses recommendation quality by simulating user preferences
   - Extracts preference tags from user profiles
   - Generates "ground truth" recommendations for comparison
   - Calculates quality metrics

3. **Prompt-Optimizer Agent**: Improves recommendation prompts based on evaluation feedback
   - Analyzes successful and unsuccessful recommendation cases
   - Proposes prompt tweaks to improve performance
   - Maintains a cache of successful prompts

The system operates in an optimization loop: recommend → evaluate → optimize → repeat.

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- API keys for OpenAI and/or Google Gemini (based on which models you plan to use)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ShuhangGe/recommendation-agents.git
   cd recommendation-agents
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys (see `.env.example`):
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## ⚙️ Configuration

The system is configured via the `config.py` file, which includes:

- API keys (loaded from `.env`)
- Available models and their types
- Default models for each task
- File paths and cache settings
- Evaluation metrics
- Optimization parameters

You can override configuration options via environment variables or command-line arguments.

### Environment Variables

Examples:
```
MAX_ITERATIONS=15
SCORE_THRESHOLD=0.85
RECOMMENDATION_MODEL=gpt-4
```

## 🚀 Usage Guide

### Basic Usage

1. **Generate Story Data**:
   ```bash
   ./generate_stories.sh --model gpt-3.5-turbo --count 100
   ```

2. **Run a Recommendation Test**:
   ```bash
   python run_recommendation.py --user-id user1 --recommendation-model gpt-3.5-turbo
   ```

3. **Run an Optimization Loop**:
   ```bash
   ./run_optimization.sh --recommendation-model gpt-3.5-turbo --evaluation-model gpt-4
   ```

4. **Test a Specific Task with Different Models**:
   ```bash
   python test_task.py --task recommendation --model gpt-4 --user-id user1
   ```

5. **Compare Multiple Models**:
   ```bash
   ./compare_models.sh --task recommendation --models "gpt-3.5-turbo,gpt-4,gemini-1.5-flash"
   ```

### Advanced Usage

For more control, use the Python API directly:

```python
from agents import RecommendationAgent, EvaluationAgent
import utils

# Load data
stories, users = utils.load_seed_data()
all_stories = utils.expand_stories(stories, target_count=100)

# Initialize agents with specific models
rec_agent = RecommendationAgent(all_stories, model_name="gpt-4")
eval_agent = EvaluationAgent(all_stories, model_name="gemini-1.5-pro")

# Extract user preferences
user_profile = users[0]['profile']
tags = eval_agent.extract_tags(user_profile)

# Generate recommendations
recommendations = rec_agent.recommend(tags)

# Get full story details
recommended_stories = rec_agent.get_stories_by_ids(recommendations)
```

## 📜 Scripts Explained

The repository includes several utility scripts:

1. **`generate_stories.sh`**: Generates synthetic story data
   - `--model`: Model to use for generation
   - `--count`: Number of stories to generate
   - `--regenerate`: Force regeneration of existing data

2. **`run_recommendation.py`**: Tests recommendations for a specific user
   - `--user-id`: ID of the user to test
   - `--recommendation-model`: Model for story recommendations
   - `--evaluation-model`: Model for evaluations
   - `--metric`: Evaluation metric to use

3. **`run_optimization.sh`**: Runs a complete optimization cycle
   - `--recommendation-model`: Model for recommendations
   - `--evaluation-model`: Model for evaluations
   - `--optimizer-model`: Model for optimization
   - `--iterations`: Maximum iterations

4. **`test_task.py`**: Tests a specific task with a specific model
   - `--task`: Task to test (recommendation, evaluation, tags, optimization)
   - `--model`: Model to use for the task
   - `--user-id`: User ID to test with

5. **`compare_models.sh`**: Compares different models for a specific task
   - `--task`: Task to test
   - `--models`: Comma-separated list of models to compare
   - `--json`: Output results in JSON format

## 🤖 Available Models

### Google Models
- `gemini-1.5-flash`: Fast, efficient model for recommendations
- `gemini-1.5-pro`: Powerful model for evaluation and optimization

### OpenAI Models
- `gpt-3.5-turbo`: Balanced performance and speed
- `gpt-4`: Most powerful model, best for complex tasks
- `gpt-4-turbo`: Faster version of GPT-4 with good performance

To list all available models:
```bash
python main.py --list-models
```

## 📊 Evaluation Metrics

The system supports multiple evaluation metrics:

1. **`precision@10`** (default): Percentage of relevant recommendations in the top 10
2. **`recall`**: Percentage of ground truth items found in the recommendations
3. **`semantic_overlap`**: Semantic similarity between recommended and ground truth stories

Configure the metric via:
```bash
python run_recommendation.py --metric semantic_overlap
```

## ⚡ Performance Optimization

To optimize system performance:

1. **Model Selection**: Use faster models for recommendation tasks, premium models for optimization
   - Example: `gemini-1.5-flash` for recommendations, `gpt-4` for optimization

2. **Caching**: The system caches:
   - Story embeddings to avoid redundant computation
   - Successful prompts along with their performance metrics

3. **Stopping Rules**: The optimization process stops when:
   - The evaluation metric plateaus (improvement < 1% over 3 consecutive iterations)
   - A maximum score threshold is reached (default: 0.8)
   - The maximum number of iterations is reached (default: 10)

## 🤝 Contributing

Contributions to the Sekai Recommendation Agent System are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's style and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 