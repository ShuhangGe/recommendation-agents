# Sekai Recommendation Agent System

This project implements a multi-agent system for recommending relevant stories to users as part of the Sekai Take-Home Challenge.

## Architecture & Agent Roles

The system consists of three main agents:

1. **Prompt-Optimizer Agent**: Proposes improvements to recommendation prompts based on evaluation feedback.
2. **Recommendation Agent**: Generates story recommendations for users based on their profile tags.
3. **Evaluation Agent**: Evaluates recommendation quality by simulating user preferences and comparing with ground truth.

These agents work in an autonomous loop: optimize → recommend → evaluate → feedback → repeat.

## Caching Strategy

- **Embedding Cache**: Story embeddings are generated once and cached to avoid redundant computation.
- **Prompt Cache**: Successful prompts are cached along with their performance metrics for future reference.

## Evaluation Metric & Stopping Rule

The system uses precision@10 as the primary metric, which measures the percentage of relevant recommendations among the top 10 suggested stories. The optimization process stops when:
- The evaluation metric plateaus (improvement less than 1% over 3 consecutive iterations)
- A maximum score threshold is reached (e.g., 0.8)
- The maximum number of iterations is reached (e.g., 10)

## Scaling to Production

To scale this system to production volumes:
1. **Distributed Processing**: Implement parallel processing for handling multiple user requests.
2. **Database Integration**: Replace in-memory storage with persistent database storage.
3. **Pre-computed Embeddings**: Generate and store embeddings for all stories in advance.
4. **Optimization Schedule**: Perform prompt optimization asynchronously on a scheduled basis.
5. **API Gateway**: Create a standardized API for integrating with front-end applications.

## Setup and Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```
4. Run the system: `python main.py`

## Results

The system demonstrates improvement over multiple optimization cycles, as documented in the `results` directory. 

## Updated Requirements

The following packages have been updated from their original versions:
- `google-generativeai==0.3.1`
- `openai==1.14.0`
- `python-dotenv==1.0.0`
- `numpy==1.24.4`
- `pandas==2.0.2`
- `scikit-learn==1.3.2`
- `matplotlib==3.7.2`
- `tqdm==4.66.3` 