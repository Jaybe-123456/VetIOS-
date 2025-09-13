import os
import json
from typing import List, Optional, Dict, Any
import logging
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from rank_bm25 import BM25Okapi
from feedback_utils import load_scores

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths - use /tmp for Render deployment
CORPUS_FILE = os.path.join("/tmp", "corpus.jsonl")
BACKUP_CORPUS_FILE = "corpus.jsonl"  # Fallback if /tmp file doesn't exist

# Configuration
DEFAULT_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL = os.environ.get("HF_EMBEDDING_MODEL", DEFAULT_HF_MODEL)
SEMANTIC_WEIGHT = 0.7  # Weight for semantic search vs BM25
BM25_WEIGHT = 0.3

def load_corpus_texts() -> tuple[List[str], List[Dict[str, Any]]]:
    """Load corpus texts from file with fallback options."""
    texts = []
    docs = []
    
    # Try loading from /tmp first, then fallback to local file
    corpus_paths = [CORPUS_FILE, BACKUP_CORPUS_FILE]
    
    for corpus_path in corpus_paths:
        if os.path.exists(corpus_path):
            try:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            rec = json.loads(line.strip())
                            
                            # Extract text with multiple fallback options
                            txt = (
                                rec.get("text") or 
                                rec.get("title") or 
                                rec.get("abstract") or 
                                rec.get("content") or
                                ""
                            )
                            
                            if txt.strip():  # Only add non-empty texts
                                texts.append(txt.strip())
                                
                                # Ensure document has required metadata
                                if "id" not in rec and "doi" not in rec and "url" not in rec:
                                    rec["id"] = f"doc_{line_num}"
                                
                                docs.append(rec)
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                            continue
                            
                logger.info(f"Loaded {len(texts)} documents from {corpus_path}")
                break
                
            except Exception as e:
                logger.error(f"Error reading corpus file {corpus_path}: {e}")
                continue
    else:
        logger.warning("No corpus file found - BM25 retrieval will be disabled")
    
    return texts, docs

class HybridRetriever:
    """
    Hybrid retriever combining semantic search (Pinecone) with lexical search (BM25)
    and user feedback-based ranking.
    """
    
    def __init__(self, pinecone_index_name: str, 
                 huggingface_model: str = HF_MODEL, 
                 pinecone_env: str = "gcp-starter"):
        """
        Initialize the hybrid retriever.
        
        Args:
            pinecone_index_name: Name of the Pinecone index
            huggingface_model: HuggingFace embedding model name
            pinecone_env: Pinecone environment
        """
        self.pinecone_index_name = pinecone_index_name
        self.hf_model = huggingface_model
        self.pinecone_env = pinecone_env
        self.semantic = None
        self.bm25 = None
        self.bm25_docs = []
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_bm25()
        self._initialize_semantic()
    
    def _initialize_embeddings(self) -> None:
        """Initialize HuggingFace embeddings."""
        try:
            self.embed = HuggingFaceEmbeddings(
                model_name=self.hf_model,
                model_kwargs={'device': 'cpu'},  # Ensure CPU usage for Render
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Initialized embeddings with model: {self.hf_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _initialize_bm25(self) -> None:
        """Initialize BM25 lexical retriever."""
        try:
            texts, docs = load_corpus_texts()
            self.bm25_docs = docs
            
            if texts:
                # Tokenize texts for BM25
                tokenized_corpus = [text.lower().split() for text in texts]
                self.bm25 = BM25Okapi(tokenized_corpus)
                logger.info(f"Initialized BM25 with {len(texts)} documents")
            else:
                logger.warning("No texts found - BM25 retrieval disabled")
                self.bm25 = None
                
        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            self.bm25 = None
    
    def _initialize_semantic(self) -> None:
        """Initialize Pinecone semantic retriever."""
        try:
            # Check if index exists
            existing_indexes = pinecone.list_indexes()
            if self.pinecone_index_name not in existing_indexes:
                logger.warning(f"Pinecone index '{self.pinecone_index_name}' not found")
                return
                
            self.semantic = Pinecone.from_existing_index(
                self.pinecone_index_name, 
                embedding=self.embed
            ).as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}  # Get more candidates for better ranking
            )
            logger.info(f"Initialized semantic retriever with index: {self.pinecone_index_name}")
            
        except Exception as e:
            logger.warning(f"Semantic retriever not available: {e}")
            self.semantic = None
    
    def _get_bm25_results(self, query: str, top_k: int = 6) -> List[Document]:
        """Get results from BM25 lexical search."""
        results = []
        
        if not self.bm25 or not self.bm25_docs:
            return results
        
        try:
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores for all documents
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top scoring documents
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            for idx in top_indices:
                if idx < len(self.bm25_docs):
                    doc_data = self.bm25_docs[idx]
                    
                    # Create Document with BM25 score
                    doc = Document(
                        page_content=doc_data.get("text", ""),
                        metadata={
                            **doc_data,
                            "retrieval_method": "bm25",
                            "bm25_score": float(scores[idx])
                        }
                    )
                    results.append(doc)
            
            logger.debug(f"BM25 retrieved {len(results)} documents")
            
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {e}")
        
        return results
    
    def _get_semantic_results(self, query: str, top_k: int = 8) -> List[Document]:
        """Get results from semantic search."""
        results = []
        
        if not self.semantic:
            return results
        
        try:
            sem_docs = self.semantic.get_relevant_documents(query)
            
            # Add retrieval method metadata
            for doc in sem_docs[:top_k]:
                doc.metadata["retrieval_method"] = "semantic"
                results.append(doc)
            
            logger.debug(f"Semantic search retrieved {len(results)} documents")
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {e}")
        
        return results
    
    def _deduplicate_and_rank(self, documents: List[Document], top_k: int) -> List[Document]:
        """Remove duplicates and rank documents using feedback scores."""
        scores = load_scores()
        seen = set()
        unique_docs = []
        
        for doc in documents:
            # Generate unique key for deduplication
            doc_key = (
                doc.metadata.get("doi") or 
                doc.metadata.get("url") or 
                doc.metadata.get("id") or 
                doc.page_content[:100]  # First 100 chars as fallback
            )
            
            if doc_key in seen:
                continue
            
            seen.add(doc_key)
            
            # Add feedback score boost
            feedback_score = scores.get(doc_key, 0)
            doc.metadata["feedback_score"] = feedback_score
            
            # Calculate composite score
            bm25_score = doc.metadata.get("bm25_score", 0)
            semantic_relevance = 1.0  # Default for semantic results
            
            # Combine scores with weights
            composite_score = (
                (bm25_score * BM25_WEIGHT) +
                (semantic_relevance * SEMANTIC_WEIGHT) +
                (feedback_score * 0.1)  # Small boost from user feedback
            )
            
            doc.metadata["composite_score"] = composite_score
            unique_docs.append(doc)
        
        # Sort by composite score
        unique_docs.sort(key=lambda x: x.metadata.get("composite_score", 0), reverse=True)
        
        logger.info(f"Ranked {len(unique_docs)} unique documents from {len(documents)} total")
        
        return unique_docs[:top_k]
    
    def get_relevant_documents(self, query: str, top_k: int = 6) -> List[Document]:
        """
        Retrieve relevant documents using hybrid approach.
        
        Args:
            query: Search query
            top_k: Number of documents to return
            
        Returns:
            List of ranked Document objects
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            all_results = []
            
            # Get BM25 results
            bm25_results = self._get_bm25_results(query, top_k)
            all_results.extend(bm25_results)
            
            # Get semantic results
            semantic_results = self._get_semantic_results(query, top_k)
            all_results.extend(semantic_results)
            
            # If no results from either method, return default documents
            if not all_results:
                return self._get_default_documents()
            
            # Deduplicate and rank
            final_results = self._deduplicate_and_rank(all_results, top_k)
            
            logger.info(f"Retrieved {len(final_results)} documents for query: '{query[:50]}...'")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return self._get_default_documents()
    
    def _get_default_documents(self) -> List[Document]:
        """Return default veterinary documents when retrieval fails."""
        default_docs = [
            Document(
                page_content="VetIOS is a veterinary AI assistant. Please ensure your knowledge base is initialized with veterinary documents.",
                metadata={
                    "title": "VetIOS Information",
                    "source": "system",
                    "retrieval_method": "fallback"
                }
            )
        ]
        return default_docs
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of retrieval components."""
        return {
            "embeddings_available": self.embed is not None,
            "bm25_available": self.bm25 is not None,
            "bm25_docs_count": len(self.bm25_docs),
            "semantic_available": self.semantic is not None,
            "pinecone_index": self.pinecone_index_name,
            "embedding_model": self.hf_model
        }
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        feedback_scores = load_scores()
        
        return {
            "total_corpus_docs": len(self.bm25_docs),
            "feedback_scores_count": len(feedback_scores),
            "average_feedback_score": (
                sum(feedback_scores.values()) / len(feedback_scores) 
                if feedback_scores else 0
            ),
            "components": {
                "bm25_enabled": self.bm25 is not None,
                "semantic_enabled": self.semantic is not None,
                "feedback_enabled": len(feedback_scores) > 0
            }
              }
