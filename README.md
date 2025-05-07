# Sekai Agent-Based Recommendation System

A sophisticated multi-agent recommendation system for interactive stories that leverages both Google's Gemini and OpenAI models, built on an advanced agent-based architecture.

## 📋 Table of Contents

- [Overview](#overview)
- [Agent-Based Architecture](#agent-based-architecture)
- [Pipeline Explanation](#pipeline-explanation)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Agents Explained](#agents-explained)
- [Core Components](#core-components)
- [Command-Line Options](#command-line-options)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Comparing with Workflow-Based Approach](#comparing-with-workflow-based-approach)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

The Sekai Agent-Based Recommendation System represents an evolution in recommendation systems, using an advanced agent-based architecture. Unlike traditional recommendation systems that follow fixed workflows, this system employs intelligent agents with perception, reasoning, and action capabilities to dynamically adapt to user preferences and context.

Key capabilities:
- Personalized story recommendations with dynamic strategy selection
- Advanced perception of user preferences (both explicit and implicit)
- Intelligent reasoning about recommendation strategies
- Detailed explanation of recommendation decisions
- Comprehensive quality evaluation with actionable feedback
- Adaptive prompt optimization for continuous improvement

## 🤖 Agent-Based Architecture

The system is built on a core agent framework that provides fundamental agent capabilities:

```
┌─────────────────────────────────────┐
│            Agent Core               │
│  ┌─────────┐ ┌──────┐ ┌──────────┐  │
│  │Perception│ │Reason│ │  Action  │  │
│  └─────────┘ └──────┘ └──────────┘  │
│  ┌─────────────────────────────────┐│
│  │             Memory              ││
│  │ ┌─────────┐      ┌───────────┐ ││
│  │ │Short-term│      │Long-term  │ ││
│  │ └─────────┘      └───────────┘ ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

This architecture is implemented with three key specialized agents:

1. **Adaptive Recommendation Agent**: Dynamically selects the most appropriate recommendation strategy based on user profile analysis
   - Different strategies: similarity search, tag-based search, hybrid search, diversity boost
   - Explained recommendations with context-appropriate detail levels

2. **Adaptive Evaluation Agent**: Intelligently assesses recommendation quality using multiple metrics
   - Analyzes recommendation strengths and weaknesses
   - Provides actionable feedback for improvement
   - Adapts evaluation strategy to user profile complexity

3. **Adaptive Optimizer Agent**: Analyzes evaluation results to optimize recommendation prompts
   - Identifies patterns in successful and unsuccessful recommendations
   - Generates improved prompts based on evaluation feedback
   - Adapts optimization strategy based on performance trends

Together, these agents form a comprehensive recommendation system that can understand user preferences, generate relevant recommendations, evaluate their quality, and continuously improve over time.

## 🔄 Pipeline Explanation

The Sekai Agent-Based Recommendation System follows a sophisticated pipeline that orchestrates the interactions between the three specialized agents:

### Initial Setup and Data Loading

The pipeline begins with:
1. Loading API keys for language models (OpenAI/Google)
2. Creating necessary directories for data, results, and logs
3. Loading and expanding seed story data and user profiles
4. Setting up comprehensive logging infrastructure

### Three-Agent Workflow

The core recommendation workflow involves these stages:

#### Stage 1: Recommendation Generation
1. **Extract User Preferences**: The evaluation agent extracts explicit and implicit preferences from user profiles
2. **Generate Recommendations**: The recommendation agent:
   - Perceives user context through profile analysis
   - Reasons about the optimal strategy (similarity search, tag-based, hybrid, etc.)
   - Executes the chosen strategy to produce personalized recommendations

#### Stage 2: Quality Evaluation
1. **Evaluate Recommendations**: The evaluation agent:
   - Analyzes profile complexity and recommendation diversity
   - Selects appropriate metrics (precision@10, recall, semantic overlap)
   - Generates "ground truth" recommendations for comparison
   - Calculates quality scores and identifies strengths/weaknesses

#### Stage 3: Prompt Optimization
1. **Optimize Recommendations**: The optimizer agent:
   - Analyzes patterns in successful and unsuccessful recommendations
   - Selects an optimization strategy based on score trends
   - Generates an improved prompt for the recommendation agent
   - Stores successful prompts in cache for future use

### Optimization Loop

The system runs an iterative optimization process:

1. For each iteration:
   - Process each test user through all three agents
   - Calculate average score across all users
   - Generate an improved prompt based on evaluation results
   - Update the recommendation agent with the new prompt

2. The loop continues until:
   - Maximum iterations are reached
   - Score threshold is exceeded
   - Improvement falls below the minimum threshold

3. The system tracks:
   - Score progression over iterations
   - Prompt changes and their impact
   - User-specific performance metrics

### Output and Analysis

The system produces:
1. Timestamped logs of the entire process
2. CSV and JSON files with detailed performance metrics
3. Visualizations of optimization progress
4. Comprehensive quality analysis for each user

This pipeline combines the strengths of all three agents to create a system that continuously improves its recommendation capability through an adaptive, intelligent process.

## ✨ Features

- **Adaptive Strategy Selection**: Dynamically chooses the best recommendation strategy based on user preferences
- **Multi-faceted User Analysis**: Extracts explicit and infers implicit preferences from user profiles
- **Memory System**: Short-term and long-term memory for tracking context across interactions
- **Tool System**: Specialized tools for different recommendation, evaluation, and optimization tasks
- **Explanation System**: Generate context-appropriate explanations for recommendations
- **Quality Analysis**: Detailed analysis of recommendation strengths and weaknesses
- **Prompt Optimization**: Continuous improvement through iterative prompt refinement
- **Dynamic Diversity**: Adjustable diversity to balance relevance with exploration
- **Performance Insights**: Detailed metrics on recommendation quality
- **Comprehensive Logging**: Real-time tracking of optimization progress with detailed analytics

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- API keys for OpenAI and/or Google Gemini (based on which models you plan to use)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ShuhangGe/recommendation-agents.git
   cd recommendation-agents
   git checkout agent-based-system
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

## 🚀 Usage

### Generating Story Data

Before running the recommendation system, you need to generate story data:

```bash
./generate_stories.sh --model gpt-4o-mini --count 100 --regenerate
```

This will generate 100 stories using the specified model and save them to the data directory.

### Running the Agent-Based Recommendation System

Run the main script to execute the agent-based recommendation system:

```bash
python main.py --recommendation-model gpt-4o-mini --evaluation-model gpt-4o-mini --optimizer-model gpt-4o-mini --verbose --enable-optimization
```

### Using the Run Script

Alternatively, use the provided shell script:

```bash
./run_agent_system.sh --recommendation-model gpt-4o-mini --evaluation-model gpt-4o-mini --optimizer-model gpt-4o-mini --verbose --enable-optimization
```

## 🧠 Agents Explained

### Agent Core

The `Agent` class provides the foundation for all agents with:

- **Perception**: Process input data to understand the environment
- **Reasoning**: Determine the best course of action based on perceptions
- **Action**: Execute the chosen strategy to achieve goals
- **Memory**: Store and retrieve information from short and long-term memory
- **Tools**: Specialized functions that agents can use for specific tasks
- **Goals**: Explicit objectives that guide the agent's behavior

### Adaptive Recommendation Agent

The recommendation agent:

1. **Perceives** user preferences by:
   - Extracting explicit preferences from tags
   - Inferring implicit preferences from profile text
   - Analyzing preference patterns from past interactions
   - Determining preference strengths

2. **Reasons** about the best recommendation strategy:
   - Chooses between similarity search, tag-based search, hybrid search
   - Decides whether to prioritize diversity
   - Selects appropriate explanation mode
   - Weighs different preferences

3. **Acts** by:
   - Executing the chosen recommendation strategy
   - Generating personalized recommendations
   - Creating context-appropriate explanations
   - Returning detailed results

### Adaptive Evaluation Agent

The evaluation agent:

1. **Perceives** the evaluation context by:
   - Analyzing user profile complexity
   - Examining recommendation diversity
   - Assessing tag relevance to recommendations

2. **Reasons** about the best evaluation approach:
   - Selects appropriate metrics based on context
   - Determines whether to use weighted scoring
   - Decides level of detail for quality analysis

3. **Acts** by:
   - Calculating multiple quality metrics
   - Analyzing recommendation strengths and weaknesses
   - Generating actionable feedback
   - Providing detailed evaluation results

### Adaptive Optimizer Agent

The optimizer agent:

1. **Perceives** evaluation results by:
   - Analyzing patterns in successful and unsuccessful recommendations
   - Identifying strengths and weaknesses in current prompts
   - Tracking performance trends across iterations

2. **Reasons** about the best optimization approach:
   - Selects appropriate optimization strategy based on trends
   - Determines which aspects of the prompt to modify
   - Decides on the degree of change needed

3. **Acts** by:
   - Generating improved prompts based on analysis
   - Providing detailed optimization rationale
   - Tracking performance across iterations
   - Adapting strategies when hitting performance plateaus

## 🧩 Core Components

### Agent Memory

Each agent has a memory system with:
- **Short-term memory**: For temporary, session-based information
- **Long-term memory**: For persistent information across sessions
- **Context history**: For tracking conversation and action history

### Agent Tools

Agents have specialized tools for different tasks:
- **Recommendation tools**: similarity_search, tag_based_search, hybrid_search, diversity_boost
- **Evaluation tools**: extract_preferences, generate_ground_truth, evaluate_precision, analyze_quality
- **Optimization tools**: analyze_results, generate_improved_prompt, detect_performance_plateau

### Perception-Reasoning-Action Loop

The core execution model follows a perception-reasoning-action loop:
1. **Perceive**: Process input data to update understanding
2. **Reason**: Determine the best strategy based on perceptions
3. **Act**: Execute the chosen strategy to achieve goals

## 💻 Command-Line Options

### Main Script Options

```
python main.py [options]
```

Available options:
- `--save-results`: Save results to a file
- `--verbose`: Print detailed information
- `--list-models`: List all available models and exit
- `--recommendation-model MODEL`: Model for recommendations
- `--evaluation-model MODEL`: Model for evaluations
- `--optimizer-model MODEL`: Model for prompt optimization
- `--story-model MODEL`: Model for generating stories
- `--user-model MODEL`: Model for generating user profiles
- `--story-count NUM`: Number of stories to generate (default: 100)
- `--user-count NUM`: Number of additional test users to generate (default: 5)
- `--regenerate-data`: Force regeneration of stories and users
- `--enable-optimization`: Enable prompt optimization loop
- `--max-iterations NUM`: Maximum number of optimization iterations (default: 20)
- `--score-threshold NUM`: Score threshold to stop optimization (default: 0.8)
- `--improvement-threshold NUM`: Minimum improvement to continue optimization (default: 0.005)
- `--metric METRIC`: Evaluation metric to use (default: precision@10)

### Agent System Script Options

```
./run_agent_system.sh [options]
```

Available options:
- `--recommendation-model MODEL`: Model for recommendations
- `--evaluation-model MODEL`: Model for evaluations
- `--optimizer-model MODEL`: Model for prompt optimization
- `--user-id USER_ID`: User ID to test (e.g., user1)
- `--profile TEXT`: User profile text (alternative to user-id)
- `--verbose`: Print detailed information about the process
- `--json`: Output results in JSON format
- `--enable-optimization`: Enable prompt optimization loop
- `--max-iterations NUM`: Maximum number of optimization iterations
- `--score-threshold NUM`: Score threshold to stop optimization
- `--improvement-threshold NUM`: Minimum improvement to continue optimization

### Story Generation Script Options

```
./generate_stories.sh [options]
```

Available options:
- `--model MODEL`: Model to use for generation
- `--count COUNT`: Number of stories to generate
- `--regenerate`: Force regeneration of existing data

## 🤖 Models

The system supports both Google Gemini and OpenAI models:

### Google Models
- `gemini-1.5-flash`: Fast, efficient model for recommendations
- `gemini-1.5-pro`: Powerful model for evaluation and complex reasoning

### OpenAI Models
- `gpt-3.5-turbo`: Balanced performance and speed
- `gpt-4`: Most powerful model for complex reasoning
- `gpt-4o-mini`: Fast and efficient for most recommendation tasks

To list all available models:
```bash
python main.py --list-models
```

## 📊 Evaluation Metrics

The system supports multiple evaluation metrics:

1. **`precision@10`** (default): Percentage of relevant recommendations in the top 10
2. **`recall`**: Percentage of ground truth items found in the recommendations
3. **`semantic_overlap`**: Semantic similarity between recommended and ground truth stories

## 🔄 Comparing with Workflow-Based Approach

The agent-based system offers several advantages over the traditional workflow-based approach:

### Architecture

| Agent-Based | Workflow-Based |
|-------------|---------------|
| Dynamic strategy selection | Fixed recommendation approach |
| Perception-reasoning-action loop | Linear processing pipeline |
| Rich memory system | Limited context tracking |
| Explicit reasoning about approaches | Implicit in code logic |

### Capabilities

| Agent-Based | Workflow-Based |
|-------------|---------------|
| Multiple recommendation strategies | Single prompt-based approach |
| Detailed explanation of recommendations | Limited explanation capabilities |
| Adaptive to user profile complexity | One-size-fits-all approach |
| Explicit goals and reasoning | Hardcoded logic |
| Continuous prompt optimization | Manual prompt engineering |

### Performance

Agent-based systems generally provide:
- More personalized recommendations
- Better handling of complex user profiles
- More detailed quality analysis
- More transparent decision-making
- Improved performance through continuous optimization

## 🤝 Contributing

Contributions to the Sekai Agent-Based Recommendation System are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's style and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 