"""
Main module for the Sekai Recommendation Agent system.
Uses the agent-based architecture for intelligent recommendations with AutoPrompt-inspired optimization.
"""
import argparse
import json
import os
import time
import csv
import datetime
import logging
import shutil
import random
import traceback
from typing import List, Dict, Any

import config
import utils
from agents import AdaptiveRecommendationAgent, AdaptiveEvaluationAgent, AdaptiveOptimizerAgent

# Global variable to store the current run directory
CURRENT_RUN_DIR = ""

def setup_directories():
    """Create necessary directories and set up a unique run directory."""
    global CURRENT_RUN_DIR
    
    # Create base directories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    
    # Create a unique run directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories in the run directory
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "json"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "edge_cases"), exist_ok=True)  # New directory for edge cases
    
    # Store the run directory path for later use
    CURRENT_RUN_DIR = run_dir
    
    # Create a README file with run information
    with open(os.path.join(run_dir, "README.txt"), 'w') as f:
        f.write(f"Optimization Run: {timestamp}\n")
        f.write("=" * 50 + "\n")
        f.write("This directory contains all logs and results for this optimization run.\n\n")
        f.write("Folder structure:\n")
        f.write("- logs/: Contains log files with detailed process information\n")
        f.write("- csv/: Contains CSV files with optimization results\n")
        f.write("- json/: Contains JSON files with detailed optimization data\n")
        f.write("- edge_cases/: Contains challenging examples used for prompt calibration\n")
    
    print(f"Created run directory: {run_dir}")
    return run_dir

def setup_logging():
    """Setup logging configuration."""
    global CURRENT_RUN_DIR
    
    # Create log directory in the run directory
    log_dir = os.path.join(CURRENT_RUN_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"optimization_log_{timestamp}.log")
    
    # Reset root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create logger
    logger = logging.getLogger("OptimizationLogger")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    
    # Create formatter for detailed file logs
    detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # Create formatter for concise console logs
    concise_formatter = logging.Formatter('%(message)s')
    
    # Create file handler with detailed logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Create console handler with more concise format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(concise_formatter)
    logger.addHandler(console_handler)
    
    # Print startup message to console
    print(f"🔄 Starting optimization run...")
    print(f"📁 Detailed logs will be saved to: {log_file}")
    print("-" * 50)
    
    return logger

def run_agent_system(args):
    """
    Run the agent-based recommendation system.
    
    Args:
        args: Command-line arguments
    """
    global CURRENT_RUN_DIR
    
    # Setup directories and get run directory
    run_dir = setup_directories()
    
    # Set up API keys
    utils.load_api_keys()
    
    # Setup logging
    logger = setup_logging()
    
    # Save command-line arguments to the run directory
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        # Convert args namespace to dict and save
        args_dict = vars(args)
        json.dump(args_dict, f, indent=2, default=str)
    
    # Log configuration - detailed version for file log
    logger.info("=== STARTING AGENT-BASED RECOMMENDATION SYSTEM ===")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Recommendation model: {args.recommendation_model}")
    logger.info(f"Evaluation model: {args.evaluation_model}")
    logger.info(f"Optimizer model: {args.optimizer_model}")
    logger.info(f"Metric: {args.metric}")
    
    # Print concise config to console
    print(f"📊 Configuration:")
    print(f"   Recommendation: {args.recommendation_model}")
    print(f"   Evaluation: {args.evaluation_model}")
    print(f"   Optimizer: {args.optimizer_model}")
    
    if args.enable_optimization:
        logger.info(f"Optimization enabled (max iterations: {args.max_iterations}, "
                  f"score threshold: {args.score_threshold}, "
                  f"improvement threshold: {args.improvement_threshold})")
        print(f"🔄 Optimization enabled:")
        print(f"   Max iterations: {args.max_iterations}")
        print(f"   Score threshold: {args.score_threshold}")
        print(f"   Min improvement: {args.improvement_threshold}")
    
    # Check if data files exist
    story_file_path = config.STORY_DATA_FILE
    user_file_path = config.USER_PROFILES_FILE
    
    if not os.path.exists(story_file_path) or not os.path.exists(user_file_path):
        error_msg = (
            "Required data files not found. Please generate data first using:\n"
            "./generate_data.sh\n\n"
            "Data files expected at:\n"
            f"- {story_file_path}\n"
            f"- {user_file_path}"
        )
        logger.error(error_msg)
        print(f"\n❌ Error: {error_msg}")
        return {"error": "Data files not found"}
    
    # Load pre-generated story and user data using utility functions
    try:
        all_stories = utils.load_stories()
        all_users = utils.load_user_profiles()
            
        logger.info(f"Loaded {len(all_stories)} stories and {len(all_users)} users from disk")
        print(f"📚 Loaded {len(all_stories)} stories and {len(all_users)} users")
    except Exception as e:
        logger.error(f"Error loading data files: {str(e)}")
        print(f"\n❌ Error: Failed to load data files: {str(e)}")
        return {"error": f"Failed to load data: {str(e)}"}
    
    logger.info(f"Testing with {len(all_users)} user(s)")
    
    # Initialize agents with specified models
    recommendation_agent = AdaptiveRecommendationAgent(all_stories, model_name=args.recommendation_model)
    evaluation_agent = AdaptiveEvaluationAgent(all_stories, model_name=args.evaluation_model)
    optimizer_agent = AdaptiveOptimizerAgent(model_name=args.optimizer_model)
    
    # Set goals for agents
    recommendation_agent.set_goal("Provide highly relevant story recommendations to users")
    recommendation_agent.set_goal("Adapt recommendation strategy based on user preferences")
    recommendation_agent.set_goal("Explain the recommendation process")
    
    evaluation_agent.set_goal("Accurately assess recommendation quality")
    evaluation_agent.set_goal("Identify strengths and weaknesses in recommendations")
    evaluation_agent.set_goal("Provide actionable feedback for improvement")
    
    optimizer_agent.set_goal("Analyze evaluation results to identify optimization opportunities")
    optimizer_agent.set_goal("Generate improved prompts for the recommendation agent")
    optimizer_agent.set_goal("Adapt optimization strategy based on past iterations")
    
    # Initialize optimization tracking
    optimization_results = []
    current_iteration = 0
    best_score = 0.0
    best_prompt = recommendation_agent.get_current_prompt()
    edge_cases_collection = []
    
    # Create real-time optimization log file in the run directory
    realtime_log_file = os.path.join(run_dir, "csv", "optimization_realtime.csv")
    
    with open(realtime_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Iteration", "Score", "Improvement", "Best Score", 
                         "Strategy", "Prompt Length", "User Scores", "Edge Cases"])
    
    # Save initial prompts
    initial_prompts_file = os.path.join(run_dir, "json", "initial_prompts.json")
    with open(initial_prompts_file, 'w') as f:
        json.dump({
            "recommendation_prompt": recommendation_agent.get_current_prompt(),
            "timestamp": time.time()
        }, f, indent=2)
    
    # Run optimization loop if enabled
    if args.enable_optimization:
        logger.info("\n=== Starting Optimization Loop ===")
        logger.info(f"Max iterations: {args.max_iterations}")
        logger.info(f"Score threshold: {args.score_threshold}")
        logger.info(f"Improvement threshold: {args.improvement_threshold}")
        
        print("\n🚀 Starting Optimization Loop...")
        
        # Optimization loop
        while current_iteration < args.max_iterations:
            current_iteration += 1
            logger.info(f"\n--- Optimization Iteration {current_iteration}/{args.max_iterations} ---")
            print(f"\n⏳ Iteration {current_iteration}/{args.max_iterations}")
            
            # Create iteration directory for this iteration's results
            iteration_dir = os.path.join(run_dir, f"iteration_{current_iteration:03d}")
            os.makedirs(iteration_dir, exist_ok=True)
            
            # Run tests for each user
            iteration_results = []
            total_score = 0.0
            user_scores = {}
            edge_cases = []
            error_counts = {"recommendation": 0, "evaluation": 0, "total": 0}
            
            # Create user results directory
            users_dir = os.path.join(iteration_dir, "users")
            os.makedirs(users_dir, exist_ok=True)
            
            for user in all_users:
                user_id = user['id']
                logger.info(f"\nProcessing user {user_id}...")
                print(f"   👤 Processing user {user_id}...")
                user_profile = user['profile']
                
                # Step 1: Extract preferences
                extracted_tags = evaluation_agent._extract_preferences(user_profile)
                
                # Step 2: Prepare input for recommendation agent
                recommendation_input = {
                    "user_id": user_id,
                    "preferences": extracted_tags,
                    "profile": user_profile,
                    "past_interactions": []
                }
                
                # Step 3: Run recommendation cycle
                recommendation_status = "pending"
                try:
                    rec_result = recommendation_agent.run_perception_reasoning_action_loop(recommendation_input)
                    recommended_stories = rec_result.get("recommended_stories", [])
                    strategy_used = rec_result.get("strategy_used", "unknown")
                    
                    if not recommended_stories:
                        logger.warning(f"No recommendations generated for user {user_id}. Using fallback.")
                        # Fallback to a simple recommendation if needed
                        recommended_stories = random.sample(all_stories, min(10, len(all_stories)))
                        strategy_used = "random_fallback"
                        
                    recommendation_status = "success"
                except Exception as e:
                    logger.error(f"Error in recommendation cycle for user {user_id}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    print(f"      ❌ Error generating recommendations: {str(e)}")
                    
                    # Fallback to a simple recommendation
                    recommended_stories = random.sample(all_stories, min(10, len(all_stories)))
                    strategy_used = "error_fallback"
                    recommendation_status = "error"
                    error_counts["recommendation"] += 1
                    error_counts["total"] += 1
                
                # Step 4: Evaluate recommendations
                evaluation_status = "pending"
                try:
                    evaluation_input = {
                        "user_id": user_id,
                        "user_profile": user_profile,
                        "recommended_stories": recommended_stories,
                        "extracted_tags": extracted_tags,
                        "metric": args.metric
                    }
                    
                    eval_result = evaluation_agent.run_perception_reasoning_action_loop(evaluation_input)
                    score = eval_result.get("score", 0.0)
                    total_score += score
                    user_scores[user_id] = score
                    evaluation_status = "success"
                    
                    # Identify edge cases (low-scoring results for prompt calibration)
                    if score < 0.5:  # Consider low scores as edge cases
                        edge_case = {
                            "user_id": user_id,
                            "user_profile": user_profile, 
                            "extracted_tags": extracted_tags,
                            "recommended_stories": [story["id"] for story in recommended_stories[:5]],
                            "score": score,
                            "iteration": current_iteration,
                            "weaknesses": eval_result.get("quality_analysis", {}).get("weaknesses", []),
                            "ground_truth": eval_result.get("ground_truth_ids", [])
                        }
                        edge_cases.append(edge_case)
                        edge_cases_collection.append(edge_case)
                except Exception as e:
                    logger.error(f"Error in evaluation cycle for user {user_id}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    print(f"      ❌ Error evaluating recommendations: {str(e)}")
                    
                    # Default score for error cases
                    score = 0.0
                    total_score += score
                    user_scores[user_id] = score
                    evaluation_status = "error"
                    error_counts["evaluation"] += 1
                    error_counts["total"] += 1
                
                # Store user results
                user_result = {
                    "user_id": user_id,
                    "score": score,
                    "strategy": strategy_used,
                    "strengths": eval_result.get("quality_analysis", {}).get("strengths", []) if evaluation_status == "success" else [],
                    "weaknesses": eval_result.get("quality_analysis", {}).get("weaknesses", []) if evaluation_status == "success" else [],
                    "scores_by_metric": eval_result.get("scores", {}) if evaluation_status == "success" else {},
                    "recommended_stories": recommended_stories,
                    "recommendation_status": recommendation_status,
                    "evaluation_status": evaluation_status,
                    "error": recommendation_status == "error" or evaluation_status == "error"
                }
                
                # Save user result to JSON
                user_result_file = os.path.join(users_dir, f"user_{user_id}.json")
                with open(user_result_file, 'w') as f:
                    json.dump(user_result, f, indent=2)
                
                iteration_results.append(user_result)
                logger.info(f"User {user_id} score: {score:.4f}")
                print(f"      Score: {score:.4f}")
            
            # Calculate average score
            avg_score = total_score / len(all_users) if all_users else 0.0
            
            # Log error summary if any errors occurred
            if error_counts["total"] > 0:
                logger.warning(f"Errors encountered: {error_counts['total']} total errors ({error_counts['recommendation']} recommendation, {error_counts['evaluation']} evaluation)")
                print(f"   ⚠️ Encountered {error_counts['total']} errors during processing")
            
            # Store iteration results
            timestamp = time.time()
            iteration_data = {
                "iteration": current_iteration,
                "score": avg_score,
                "user_scores": user_scores,
                "prompt": recommendation_agent.get_current_prompt(),
                "prompt_length": len(recommendation_agent.get_current_prompt()),
                "timestamp": timestamp,
                "results": iteration_results,
                "edge_cases": edge_cases,
                "error_counts": error_counts,
                "success_rate": (len(all_users) - error_counts["total"]) / len(all_users) if len(all_users) > 0 else 0
            }
            
            # Add improvement metric if not first iteration
            if current_iteration > 1:
                prev_score = optimization_results[-1]["score"]
                improvement = avg_score - prev_score
                iteration_data["improvement"] = improvement
                logger.info(f"\nIteration {current_iteration} average score: {avg_score:.4f} (improvement: {improvement:.4f})")
                print(f"   📈 Average score: {avg_score:.4f} (change: {improvement:+.4f})")
            else:
                logger.info(f"\nIteration {current_iteration} average score: {avg_score:.4f}")
                print(f"   📈 Average score: {avg_score:.4f}")
            
            # Save edge cases to file
            if edge_cases:
                edge_cases_file = os.path.join(run_dir, "edge_cases", f"edge_cases_iteration_{current_iteration}.json")
                with open(edge_cases_file, 'w') as f:
                    json.dump(edge_cases, f, indent=2)
                print(f"   ⚠️ Found {len(edge_cases)} edge cases for prompt calibration")
            
            optimization_results.append(iteration_data)
            
            # Save iteration data
            iteration_data_file = os.path.join(iteration_dir, "iteration_data.json")
            with open(iteration_data_file, 'w') as f:
                json.dump({
                    "iteration": current_iteration,
                    "score": avg_score,
                    "user_scores": user_scores,
                    "prompt_length": len(recommendation_agent.get_current_prompt()),
                    "timestamp": timestamp,
                    "edge_case_count": len(edge_cases)
                }, f, indent=2)
            
            # Log to real-time file
            log_optimization_iteration(
                realtime_log_file, 
                iteration_data, 
                strategy_used if len(all_users) == 1 else "multiple",
                len(edge_cases)
            )
            
            # Update best score and prompt if improved
            if avg_score > best_score:
                best_score = avg_score
                best_prompt = recommendation_agent.get_current_prompt()
                
                # Save best prompt
                best_prompt_file = os.path.join(run_dir, "json", "best_prompt.json")
                with open(best_prompt_file, 'w') as f:
                    json.dump({
                        "prompt": best_prompt,
                        "score": best_score,
                        "iteration": current_iteration,
                        "timestamp": time.time()
                    }, f, indent=2)
                
                logger.info(f"New best score: {best_score:.4f}")
                print(f"   🏆 New best score!")
            
            # Check if we should stop optimization
            if avg_score >= args.score_threshold:
                logger.info(f"Reached score threshold ({args.score_threshold}). Stopping optimization.")
                print(f"   ✅ Reached score threshold ({args.score_threshold}). Stopping.")
                break
                
            if current_iteration > 1:
                prev_score = optimization_results[current_iteration-2]["score"]
                improvement = avg_score - prev_score
                if improvement < args.improvement_threshold and current_iteration > 3:
                    logger.info(f"Insufficient improvement ({improvement:.4f} < {args.improvement_threshold}). Stopping optimization.")
                    print(f"   ⚠️ Insufficient improvement ({improvement:.4f} < {args.improvement_threshold}). Stopping.")
                    break
            
            # Run optimizer agent to generate new prompt
            logger.info("\nRunning optimizer agent to generate new prompt...")
            print(f"   ⚙️ Optimizing prompt...")
            
            # Prepare input for optimizer agent
            optimizer_input = {
                "current_prompt": recommendation_agent.get_current_prompt(),
                "evaluation_results": iteration_results,
                "current_score": avg_score,
                "iteration": current_iteration,
                "edge_cases": edge_cases
            }
            
            # Run optimizer agent
            optimizer_result = optimizer_agent.run_perception_reasoning_action_loop(optimizer_input)
            optimized_prompt = optimizer_result.get("optimized_prompt")
            
            # Apply optimized prompt for next iteration
            if optimized_prompt:
                recommendation_agent.update_prompt(optimized_prompt)
                logger.info("Applied optimized prompt for next iteration.")
                print(f"   ✨ Applied optimized prompt for next iteration")
            else:
                logger.warning("Optimizer agent did not return an optimized prompt.")
                print(f"   ❌ Failed to generate optimized prompt")
        
        # Save all edge cases collected during optimization
        if edge_cases_collection:
            all_edge_cases_file = os.path.join(run_dir, "edge_cases", "all_edge_cases.json")
            with open(all_edge_cases_file, 'w') as f:
                json.dump(edge_cases_collection, f, indent=2)
            
            print(f"\n📊 Collected {len(edge_cases_collection)} edge cases across all iterations")
        
        # Save optimization results
        save_optimization_results(optimization_results, args, run_dir, logger)
            
        # Restore best prompt for final evaluation
        recommendation_agent.update_prompt(best_prompt)
        logger.info(f"\n=== Optimization Complete ===")
        logger.info(f"Best score: {best_score:.4f} (iteration {get_best_iteration(optimization_results)})")
        
        print(f"\n✅ Optimization Complete")
        print(f"   🏆 Best score: {best_score:.4f} (iteration {get_best_iteration(optimization_results)})")
    
    # Run final evaluation with all users
    logger.info("\n=== Running Final Evaluation ===")
    print(f"\n📊 Running Final Evaluation...")
    results = []
    total_score = 0.0
    
    # Create final evaluation directory
    final_eval_dir = os.path.join(run_dir, "final_evaluation")
    os.makedirs(final_eval_dir, exist_ok=True)
    os.makedirs(os.path.join(final_eval_dir, "users"), exist_ok=True)
    
    for user in all_users:
        logger.info(f"\nProcessing user {user['id']}...")
        print(f"   👤 Processing user {user['id']}...")
        user_profile = user['profile']
        
        # Step 1: Extract preferences
        extracted_tags = evaluation_agent._extract_preferences(user_profile)
        logger.info(f"Extracted tags: {extracted_tags}")
        
        # Step 2: Prepare input for recommendation agent
        recommendation_input = {
            "preferences": extracted_tags,
            "profile": user_profile,
            "past_interactions": []
        }
        
        # Step 3: Run recommendation cycle
        logger.info("\nGenerating recommendations...")
        start_time = time.time()
        rec_result = recommendation_agent.run_perception_reasoning_action_loop(recommendation_input)
        recommended_stories = rec_result.get("recommended_stories", [])
        strategy_used = rec_result.get("strategy_used", "unknown")
        logger.info(f"Used strategy: {strategy_used}")
        
        # Step 4: Evaluate recommendations
        logger.info("\nEvaluating recommendations...")
        evaluation_input = {
            "user_profile": user_profile,
            "recommended_stories": recommended_stories,
            "extracted_tags": extracted_tags,
            "metric": args.metric
        }
        
        eval_result = evaluation_agent.run_perception_reasoning_action_loop(evaluation_input)
        score = eval_result.get("score", 0.0)
        scores = eval_result.get("scores", {})
        
        total_score += score
        
        # Store detailed results
        user_result = {
            "user_id": user['id'],
            "score": score,
            "recommendation_strategy": strategy_used,
            "strengths": eval_result.get("quality_analysis", {}).get("strengths", []),
            "weaknesses": eval_result.get("quality_analysis", {}).get("weaknesses", []),
            "scores_by_metric": scores,
            "recommended_stories": recommended_stories,
            "processing_time": time.time() - start_time
        }
        
        # Save user result
        user_result_path = os.path.join(final_eval_dir, "users", f"user_{user['id']}.json")
        with open(user_result_path, 'w') as f:
            json.dump(user_result, f, indent=2)
        
        results.append(user_result)
        logger.info(f"User {user['id']} score: {score:.4f}")
        print(f"      Score: {score:.4f}")
        
        # Log score breakdown
        for metric, metric_score in scores.items():
            logger.info(f"  - {metric}: {metric_score:.4f}")
        
    # Calculate final average score
    avg_score = total_score / len(all_users) if all_users else 0.0
    logger.info(f"\nFinal average score across all users: {avg_score:.4f}")
    print(f"\n📈 Final average score: {avg_score:.4f}")
    
    # Save final evaluation summary
    summary_path = os.path.join(final_eval_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "average_score": avg_score,
            "user_count": len(all_users),
            "recommendation_model": args.recommendation_model,
            "evaluation_model": args.evaluation_model,
            "metric": args.metric,
            "timestamp": time.time(),
            "optimization_enabled": args.enable_optimization,
            "optimization_iterations": current_iteration if args.enable_optimization else 0,
            "best_optimization_score": best_score if args.enable_optimization else None,
            "edge_cases_collected": len(edge_cases_collection)
        }, f, indent=2)
    
    # Create run summary
    summary_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "final_score": avg_score,
            "user_count": len(all_users),
            "story_count": len(all_stories),
            "recommendation_model": args.recommendation_model,
            "evaluation_model": args.evaluation_model,
            "optimizer_model": args.optimizer_model if args.enable_optimization else None,
            "optimization_enabled": args.enable_optimization,
            "optimization_iterations": current_iteration if args.enable_optimization else 0,
            "best_optimization_score": best_score if args.enable_optimization else None,
            "timestamp": time.time(),
            "runtime": time.time() - start_time,
            "edge_cases_collected": len(edge_cases_collection)
        }, f, indent=2)
    
    logger.info(f"\n=== Recommendation Process Complete ===")
    print(f"\n✨ Process Complete!")
    print(f"   📁 Results saved to: {run_dir}")
    
    return {
        "results": results,
        "average_score": avg_score,
        "run_directory": run_dir
    }

def log_optimization_iteration(log_file, iteration_data, strategy_used, edge_case_count=0):
    """
    Log optimization iteration to real-time CSV file.
    
    Args:
        log_file: Path to log file
        iteration_data: Data for this iteration
        strategy_used: Strategy used for recommendations
        edge_case_count: Number of edge cases found in this iteration
    """
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format user scores as a string
        user_scores_str = ", ".join([f"{k}:{v:.4f}" for k, v in iteration_data.get("user_scores", {}).items()])
        
        # Write row
        writer.writerow([
            timestamp,
            iteration_data["iteration"],
            iteration_data["score"],
            iteration_data.get("improvement", 0.0),
            iteration_data.get("best_score_so_far", iteration_data["score"]),
            strategy_used,
            iteration_data.get("prompt_length", 0),
            user_scores_str,
            edge_case_count
        ])

def save_optimization_results(optimization_results, args, run_dir, logger):
    """Save optimization results to CSV and JSON files."""
    # Create results directories in the run directory
    csv_dir = os.path.join(run_dir, "csv")
    json_dir = os.path.join(run_dir, "json")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    # Save as CSV with detailed results
    csv_file = os.path.join(csv_dir, "optimization_results.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Iteration", "Score", "Improvement", "Best Score So Far", 
            "Prompt Length", "Prompt Change Size", "Edge Case Count", "Timestamp"
        ])
        
        prev_prompt_length = 0
        for i, result in enumerate(optimization_results):
            improvement = 0.0
            if i > 0:
                improvement = result["score"] - optimization_results[i-1]["score"]
            
            prompt_length = len(result.get("prompt", ""))
            prompt_change = prompt_length - prev_prompt_length if i > 0 else 0
            prev_prompt_length = prompt_length
            
            best_score_so_far = max([r["score"] for r in optimization_results[:i+1]])
            edge_cases = result.get("edge_cases", [])
            
            writer.writerow([
                result["iteration"], 
                result["score"], 
                improvement,
                best_score_so_far,
                prompt_length,
                prompt_change,
                len(edge_cases),
                datetime.datetime.fromtimestamp(result.get("timestamp", time.time())).strftime("%Y-%m-%d %H:%M:%S")
            ])
    
    # Save as JSON with full details
    json_file = os.path.join(json_dir, "optimization_detailed.json")
    with open(json_file, 'w') as f:
        json.dump({
            "configuration": {
                "recommendation_model": args.recommendation_model,
                "evaluation_model": args.evaluation_model,
                "optimizer_model": args.optimizer_model,
                "metric": args.metric,
                "max_iterations": args.max_iterations,
                "score_threshold": args.score_threshold,
                "improvement_threshold": args.improvement_threshold
            },
            "results": optimization_results,
            "timestamp": time.time()
        }, f, indent=2)
    
    # Create a simple scores-only CSV for easy graphing
    scores_csv_file = os.path.join(csv_dir, "scores.csv")
    with open(scores_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Score", "Improvement", "Edge Cases"])
        
        for i, result in enumerate(optimization_results):
            improvement = 0.0
            if i > 0:
                improvement = result["score"] - optimization_results[i-1]["score"]
            writer.writerow([
                result["iteration"], 
                result["score"], 
                improvement, 
                len(result.get("edge_cases", []))
            ])
    
    # Create a visualization-friendly JSON with just scores
    scores_json_file = os.path.join(json_dir, "scores.json")
    with open(scores_json_file, 'w') as f:
        scores_data = []
        for i, result in enumerate(optimization_results):
            improvement = 0.0
            if i > 0:
                improvement = result["score"] - optimization_results[i-1]["score"]
            
            scores_data.append({
                "iteration": result["iteration"],
                "score": result["score"],
                "improvement": improvement,
                "edge_case_count": len(result.get("edge_cases", [])),
                "timestamp": result.get("timestamp", 0)
            })
        
        json.dump(scores_data, f, indent=2)
    
    logger.info(f"Optimization results saved to {run_dir}/csv and {run_dir}/json directories")

def get_best_iteration(optimization_results):
    """Get the iteration with the highest score."""
    best_iteration = 1
    best_score = 0.0
    
    for result in optimization_results:
        if result["score"] > best_score:
            best_score = result["score"]
            best_iteration = result["iteration"]
    
    return best_iteration

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Sekai Agent-Based Recommendation System')
    
    # Basic configuration
    parser.add_argument('--save-results', action='store_true',
                        help='Save results to a file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    
    # Model selection
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and exit')
    parser.add_argument('--recommendation-model', type=str, default=config.RECOMMENDATION_MODEL,
                        help=f'Model for recommendations (default: {config.RECOMMENDATION_MODEL})')
    parser.add_argument('--evaluation-model', type=str, default=config.EVALUATION_MODEL,
                        help=f'Model for evaluations (default: {config.EVALUATION_MODEL})')
    parser.add_argument('--optimizer-model', type=str, default=config.OPTIMIZER_MODEL,
                        help=f'Model for prompt optimization (default: {config.OPTIMIZER_MODEL})')
    
    # Optimization settings
    parser.add_argument('--enable-optimization', action='store_true',
                        help='Enable prompt optimization loop')
    parser.add_argument('--max-iterations', type=int, default=config.MAX_ITERATIONS,
                        help=f'Maximum number of optimization iterations (default: {config.MAX_ITERATIONS})')
    parser.add_argument('--score-threshold', type=float, default=config.SCORE_THRESHOLD,
                        help=f'Score threshold to stop optimization (default: {config.SCORE_THRESHOLD})')
    parser.add_argument('--improvement-threshold', type=float, default=config.IMPROVEMENT_THRESHOLD,
                        help=f'Minimum improvement to continue optimization (default: {config.IMPROVEMENT_THRESHOLD})')
    
    # Budget constraint
    parser.add_argument('--max-budget', type=float, default=None,
                        help='Maximum budget in dollars for the optimization run (default: no limit)')
    
    # Evaluation settings
    parser.add_argument('--metric', type=str, choices=['precision@10', 'recall', 'semantic_overlap'],
                        default=config.METRIC, help=f'Evaluation metric to use (default: {config.METRIC})')
    
    args = parser.parse_args()
    
    # If the user wants to list available models, print them and exit
    if args.list_models:
        print(config.list_available_models())
        return
    
    # Run the agent system with parsed arguments
    run_agent_system(args)

if __name__ == "__main__":
    main() 