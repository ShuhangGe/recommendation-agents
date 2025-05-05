"""
Utility functions for evaluating recommendation quality.
"""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def precision_at_k(recommended_ids: List[str], ground_truth_ids: List[str], k: int = 10) -> float:
    """
    Calculate precision@k metric.
    
    Args:
        recommended_ids: List of recommended story IDs
        ground_truth_ids: List of ground truth story IDs
        k: Number of top recommendations to consider
        
    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if not recommended_ids or not ground_truth_ids:
        return 0.0
    
    # Take top-k recommendations
    top_k_recommendations = recommended_ids[:k]
    
    # Count how many of the top-k recommendations are in the ground truth
    relevant_count = sum(1 for rec_id in top_k_recommendations if rec_id in ground_truth_ids)
    
    # Calculate precision@k
    return relevant_count / min(k, len(top_k_recommendations))

def recall(recommended_ids: List[str], ground_truth_ids: List[str]) -> float:
    """
    Calculate recall metric.
    
    Args:
        recommended_ids: List of recommended story IDs
        ground_truth_ids: List of ground truth story IDs
        
    Returns:
        Recall score (0.0 to 1.0)
    """
    if not recommended_ids or not ground_truth_ids:
        return 0.0
    
    # Count how many of the ground truth items are in the recommendations
    relevant_count = sum(1 for gt_id in ground_truth_ids if gt_id in recommended_ids)
    
    # Calculate recall
    return relevant_count / len(ground_truth_ids)

def semantic_overlap(recommended_stories: List[Dict[str, Any]], ground_truth_stories: List[Dict[str, Any]]) -> float:
    """
    Calculate semantic overlap between recommended and ground truth stories.
    
    Args:
        recommended_stories: List of recommended story objects
        ground_truth_stories: List of ground truth story objects
        
    Returns:
        Semantic similarity score (0.0 to 1.0)
    """
    if not recommended_stories or not ground_truth_stories:
        return 0.0
    
    # Extract text content from stories
    recommended_texts = []
    for story in recommended_stories:
        text = f"{story['title']} {story['intro']} {' '.join(story['tags'])}"
        recommended_texts.append(text)
    
    ground_truth_texts = []
    for story in ground_truth_stories:
        text = f"{story['title']} {story['intro']} {' '.join(story['tags'])}"
        ground_truth_texts.append(text)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    all_texts = recommended_texts + ground_truth_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split the matrix back into recommendation and ground truth parts
    rec_vectors = tfidf_matrix[:len(recommended_texts)]
    gt_vectors = tfidf_matrix[len(recommended_texts):]
    
    # Calculate cosine similarity between each pair
    similarities = cosine_similarity(rec_vectors, gt_vectors)
    
    # Take the average of the maximum similarity for each recommendation
    max_similarities = np.max(similarities, axis=1)
    avg_similarity = np.mean(max_similarities)
    
    return float(avg_similarity)

def calculate_metric(recommended_stories: List[Dict[str, Any]], ground_truth_stories: List[Dict[str, Any]], 
                    metric_name: str = "precision@10") -> float:
    """
    Calculate the specified evaluation metric.
    
    Args:
        recommended_stories: List of recommended story objects
        ground_truth_stories: List of ground truth story objects
        metric_name: Name of the metric to calculate
        
    Returns:
        Metric score (0.0 to 1.0)
    """
    # Extract IDs
    recommended_ids = [story["id"] for story in recommended_stories]
    ground_truth_ids = [story["id"] for story in ground_truth_stories]
    
    if metric_name == "precision@10":
        return precision_at_k(recommended_ids, ground_truth_ids, k=10)
    elif metric_name == "recall":
        return recall(recommended_ids, ground_truth_ids)
    elif metric_name == "semantic_overlap":
        return semantic_overlap(recommended_stories, ground_truth_stories)
    else:
        raise ValueError(f"Unknown metric: {metric_name}") 