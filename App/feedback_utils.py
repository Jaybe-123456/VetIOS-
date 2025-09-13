import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use /tmp for file storage on Render (ephemeral storage)
SCORES_FILE = os.path.join("/tmp", "feedback_scores.json")
FEEDBACK_LOG_FILE = os.path.join("/tmp", "feedback_log.jsonl")

def load_scores() -> Dict[str, float]:
    """
    Load feedback scores from file.
    Returns empty dict if file doesn't exist.
    """
    try:
        if os.path.exists(SCORES_FILE):
            with open(SCORES_FILE, "r", encoding="utf-8") as f:
                scores = json.load(f)
                logger.debug(f"Loaded {len(scores)} feedback scores")
                return scores
        else:
            logger.debug("No existing scores file found, starting fresh")
            return {}
    except Exception as e:
        logger.error(f"Error loading scores: {e}")
        return {}

def save_scores(scores: Dict[str, float]) -> bool:
    """
    Save feedback scores to file.
    Returns True if successful, False otherwise.
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(SCORES_FILE), exist_ok=True)
        
        with open(SCORES_FILE, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
        
        logger.debug(f"Saved {len(scores)} feedback scores")
        return True
    except Exception as e:
        logger.error(f"Error saving scores: {e}")
        return False

def update_scores(sources: List[Dict[str, Any]], approved: bool) -> Dict[str, float]:
    """
    Update scores for sources based on user feedback.
    
    Args:
        sources: List of source documents with metadata
        approved: Whether the user approved the answer
        
    Returns:
        Updated scores dictionary
    """
    try:
        scores = load_scores()
        delta = 1 if approved else -1
        updated_count = 0
        
        for source in sources:
            # Try multiple ways to identify the document
            doc_id = (
                source.get("doi") or 
                source.get("url") or 
                source.get("id") or 
                source.get("title") or
                source.get("content", "")[:100]  # First 100 chars as fallback
            )
            
            if not doc_id or doc_id == "N/A":
                continue
                
            # Update score
            old_score = scores.get(doc_id, 0)
            scores[doc_id] = old_score + delta
            updated_count += 1
            
            logger.debug(f"Updated score for {doc_id[:50]}...: {old_score} -> {scores[doc_id]}")
        
        # Save updated scores
        if updated_count > 0:
            save_scores(scores)
            logger.info(f"Updated scores for {updated_count} sources (approved: {approved})")
        else:
            logger.warning("No valid document IDs found in sources")
            
        return scores
        
    except Exception as e:
        logger.error(f"Error updating scores: {e}")
        return load_scores()  # Return existing scores if update fails

def log_feedback(question: str, answer: str, sources: List[Dict[str, Any]], 
                approved: bool, user_comment: Optional[str] = None) -> bool:
    """
    Log detailed feedback to a file for analysis.
    
    Args:
        question: Original user question
        answer: AI-generated answer
        sources: Source documents used
        approved: Whether user approved the answer
        user_comment: Optional user comment
        
    Returns:
        True if logging successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(FEEDBACK_LOG_FILE), exist_ok=True)
        
        feedback_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": answer,
            "sources_count": len(sources),
            "approved": approved,
            "user_comment": user_comment,
            "sources": [
                {
                    "id": source.get("doi") or source.get("url") or source.get("title", "unknown"),
                    "title": source.get("title", "No title"),
                    "url": source.get("url", "N/A"),
                    "category": source.get("category", "general"),
                    "species": source.get("species", "unknown")
                }
                for source in sources
            ]
        }
        
        with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry) + "\n")
            
        logger.info(f"Logged feedback entry (approved: {approved})")
        return True
        
    except Exception as e:
        logger.error(f"Error logging feedback: {e}")
        return False

def get_score_stats() -> Dict[str, Any]:
    """
    Get statistics about feedback scores.
    
    Returns:
        Dictionary with score statistics
    """
    try:
        scores = load_scores()
        
        if not scores:
            return {
                "total_documents": 0,
                "average_score": 0,
                "positive_scores": 0,
                "negative_scores": 0,
                "neutral_scores": 0
            }
        
        values = list(scores.values())
        positive_scores = len([s for s in values if s > 0])
        negative_scores = len([s for s in values if s < 0])
        neutral_scores = len([s for s in values if s == 0])
        
        return {
            "total_documents": len(scores),
            "average_score": sum(values) / len(values),
            "max_score": max(values),
            "min_score": min(values),
            "positive_scores": positive_scores,
            "negative_scores": negative_scores,
            "neutral_scores": neutral_scores
        }
        
    except Exception as e:
        logger.error(f"Error calculating score stats: {e}")
        return {"error": str(e)}

def get_top_sources(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get top-rated sources based on feedback scores.
    
    Args:
        limit: Maximum number of sources to return
        
    Returns:
        List of top sources with their scores
    """
    try:
        scores = load_scores()
        
        # Sort by score (highest first)
        sorted_sources = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"document_id": doc_id, "score": score}
            for doc_id, score in sorted_sources[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Error getting top sources: {e}")
        return []
