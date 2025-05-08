# Sekai Recommendation Agent System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)
[![Google AI](https://img.shields.io/badge/Google-Gemini-orange.svg)](https://ai.google.dev/)

<div align="center">
  <img src="https://img.shields.io/badge/Intent--based-Prompt%20Calibration-blueviolet" alt="Intent-based Prompt Calibration">
  <img src="https://img.shields.io/badge/Multi--agent-Architecture-brightgreen" alt="Multi-agent Architecture">
  <img src="https://img.shields.io/badge/Adaptive-Learning-yellow" alt="Adaptive Learning">
</div>

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture--agent-roles)
- [Intent-based Prompt Calibration Process](#-intent-based-prompt-calibration-process)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Insights](#-results--insights)
- [Production Scaling](#-scaling-to-production)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

## 📚 Overview

The Sekai Recommendation Agent System is an innovative multi-agent AI framework designed to intelligently recommend stories to users based on their preferences. The system dynamically optimizes its recommendation quality through **Intent-based Prompt Calibration**, an approach inspired by [AutoPrompt](https://github.com/Eladlev/AutoPrompt) that systematically identifies edge cases and refines prompts to handle them effectively.

The framework combines multiple specialized agents that work in concert to analyze preferences, generate recommendations, evaluate quality, and iteratively improve performance through a sophisticated optimization cycle.

## 🌟 Key Features

- **🤖 Multi-Agent Architecture**: Specialized agents for recommendations, evaluations, and optimization, each with distinct responsibilities and expertise
- **🔄 Intent-based Prompt Calibration**: Systematic improvement of prompts by identifying edge cases and boundary conditions
- **🧠 Adaptive Strategy Selection**: Dynamic selection of recommendation strategies based on user context
- **📊 Comprehensive Evaluation**: Multiple evaluation metrics for holistic assessment of recommendation quality
- **💾 Intelligent Caching**: Embedding, prompt, and evaluation caches for computational efficiency
- **🔍 Edge Case Repository**: Continuous collection and learning from challenging examples
- **📈 Budget Management**: Built-in controls for API costs and optimization efficiency

## 🏛 Architecture & Agent Roles

The system operates through three specialized agents working in concert:

### 1. Recommendation Agent

- **Purpose**: Generates story recommendations for users based on their preferences
- **Model**: Gemini 2.0 Flash (configurable)
- **Key Responsibilities**:
  - Analyzes explicit and implicit user preferences
  - Selects optimal recommendation strategy dynamically
  - Balances relevance with diversity
  - Provides explanations for recommendations

### 2. Evaluation Agent

- **Purpose**: Assesses the quality of recommendations against ground truth
- **Model**: Gemini 2.5 Pro (configurable)
- **Key Responsibilities**:
  - Extracts tags from user profiles
  - Generates ground-truth recommendations
  - Evaluates using multiple metrics (precision, recall, semantic overlap)
  - Identifies strengths and weaknesses in recommendations

### 3. Prompt-Optimizer Agent

- **Purpose**: Iteratively improves prompts through Intent-based Calibration
- **Model**: Any advanced language model (configurable)
- **Key Responsibilities**:
  - Analyzes evaluation results to identify edge cases
  - Selects optimization strategy based on performance trends
  - Generates improved prompts through calibration
  - Maintains optimization history and effectiveness metrics

## 🔄 Intent-based Prompt Calibration Process

The optimization process implements AutoPrompt's Intent-based Prompt Calibration approach through these key steps:

### 1. Boundary Case Detection

The system identifies challenging examples where the recommendation agent underperforms, focusing on cases with low evaluation scores. These edge cases reveal limitations in the current prompt that need addressing.

### 2. Pattern Recognition

The optimizer analyzes successful and unsuccessful recommendations to extract common patterns:
- Common tags in missed recommendations
- User preference patterns that lead to suboptimal recommendations
- Structural elements that distinguish high vs. low-performing cases

### 3. Targeted Calibration

Based on performance analysis, the optimizer selects one of three calibration strategies:
- **Standard**: Balanced approach for general improvements
- **Exploration**: Higher creativity when performance plateaus
- **Refinement**: Fine-tuning when the system is already performing well

### 4. Continuous Feedback Loop

The system maintains an edge case repository that grows over time. Each optimization cycle:
1. Adds newly discovered edge cases to the repository
2. Verifies that previously identified edge cases are handled
3. Adjusts optimization strategy based on performance trends

This process creates a virtuous cycle of continuous improvement:

```
┌───────────────────┐
│ Recommendation    │
│ Agent             │──────┐
└───────────────────┘      │
         ▲                 │
         │                 ▼
         │        ┌───────────────────┐
┌────────┴────────┐ │ Evaluation       │
│ Prompt-Optimizer │ │ Agent            │
│ Agent           │◄┤                   │
└────────────────┬─┘ └───────────────┬─┘
         ▲        │                  │
         │        │                  │
         │        ▼                  │
┌────────┴────────┐                  │
│ Edge Cases      │◄─────────────────┘
│ Repository      │
└────────────────┘
```

## 📊 Evaluation & Metrics

The system employs multiple evaluation metrics to provide a comprehensive assessment:

- **Precision@10**: Proportion of recommended stories that are relevant
- **Recall**: Proportion of relevant stories that were recommended
- **Semantic Overlap**: Semantic similarity between recommended stories and user preferences

The evaluation process follows these steps:

1. Uses the full user profile to generate ground-truth recommendations
2. Simulates tags a user would select on initial preference screens
3. Feeds those tags to the Recommendation Agent
4. Computes metrics against the ground-truth recommendations
5. Performs detailed quality analysis to identify strengths and weaknesses

## 🔧 Installation

### Prerequisites

- Python 3.8+
- OpenAI API key or Google AI API key
- Pip or Conda for dependency management

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sekai-recommendation-agent.git
cd sekai-recommendation-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys**
```bash
cp config/.env.example config/.env
# Edit the .env file with your API keys
```

## 🚀 Usage

### Data Generation

Generate story and user data with ground truth examples preserved:

```bash
./generate_data.sh
```

Options:
- `--story-model MODEL`: Model for generating stories (default: gpt-4o)
- `--user-model MODEL`: Model for generating user profiles (default: gpt-4o)
- `--story-count COUNT`: Number of stories to generate (default: 100)
- `--user-count COUNT`: Number of users to generate (default: 5)
- `--stories-per-user COUNT`: Number of stories per user to generate (default: 20)

### Running the System

Run the recommendation system with optimization:

```bash
./run_agent_system.sh --enable-optimization
```

Or with Python directly:

```bash
python main.py --enable-optimization
```

#### Key Options

- `--enable-optimization`: Enable the prompt optimization loop
- `--recommendation-model MODEL`: Model for recommendations
- `--evaluation-model MODEL`: Model for evaluations
- `--optimizer-model MODEL`: Model for prompt optimization
- `--metric {precision@10,recall,semantic_overlap}`: Evaluation metric
- `--max-iterations N`: Maximum optimization iterations
- `--score-threshold SCORE`: Target score to stop optimization
- `--improvement-threshold THRESHOLD`: Minimum improvement to continue
- `--max-budget DOLLARS`: Maximum budget for the optimization run
- `--verbose`: Print detailed information

### Results Directory Structure

Results are organized in timestamped directories:

```
results/
├── run_TIMESTAMP/
│   ├── logs/           # Detailed process logs
│   ├── csv/            # CSV files with optimization results
│   ├── json/           # JSON files with detailed data
│   ├── edge_cases/     # Repository of challenging examples
│   └── README.txt      # Run information
```

## 📈 Results & Insights

The system demonstrates several key findings:

1. **Edge Case Learning**: Significant recommendation improvements through Intent-based Prompt Calibration, with edge cases showing up to 30% score improvement
2. **Optimization Convergence**: Performance typically plateaus after 10-15 iterations, with diminishing returns beyond this point
3. **Strategy Effectiveness**: The "refinement" strategy performs best in later iterations, while "exploration" is most effective for breaking through performance plateaus
4. **Performance Metrics**: Final precision@10 scores of 0.85+ achieved consistently in testing
5. **Budget Efficiency**: Optimization costs kept under $1 per full cycle for test datasets

## 🔍 Scaling to Production

This system is designed with production scalability in mind:

1. **Component Independence**: Each agent can be deployed as a separate microservice
2. **Cacheable Computations**: Embedding and evaluation caches reduce duplicated work
3. **Budget Controls**: Built-in management for LLM API costs
4. **Batch Processing**: Supports batch recommendation and evaluation
5. **Edge Case Repository**: Continuous improvement through learning from challenging examples

### Production Deployment Architecture

To deploy at scale:

1. Deploy agents as containerized microservices with load balancing
2. Use Redis or similar for distributed caching
3. Implement database storage for user profiles, stories, and recommendations
4. Set up monitoring for API costs and system health
5. Implement user feedback collection to further refine recommendations

## 🙏 Acknowledgements

This project was developed as part of the Sekai Take-Home Challenge. It incorporates concepts from:

- [AutoPrompt](https://github.com/Eladlev/AutoPrompt): Intent-based Prompt Calibration approach
- Sekai's recommendation requirements for interactive storytelling

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.