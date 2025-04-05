import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import LocalOutlierFactor
import faiss
import os
from .config import config

logger = logging.getLogger(__name__)

class PhraseAnomalyDetector:
    def __init__(self, use_gpu=True):
        self.model = None
        self.vector_db = None
        self.index = None
        self.phrases = []
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.db_initialized = False
        self.embedding_dim = 768  # Default for most sentence transformers
        
    def _get_model(self):
        if self.model is None:
            logger.info("Loading sentence transformer model")
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                if self.use_gpu:
                    self.model = self.model.to('cuda')
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model: {e}")
                raise
        return self.model
    
    def _initialize_db(self):
        if not self.db_initialized:
            logger.info("Initializing vector database")
            try:
                # Create FAISS index
                self.embedding_dim = self._get_model().get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                
                # Use GPU if available and requested
                if self.use_gpu and faiss.get_num_gpus() > 0:
                    logger.info("Using GPU for FAISS")
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    
                self.db_initialized = True
                logger.info(f"Vector database initialized with dimension {self.embedding_dim}")
            except Exception as e:
                logger.error(f"Failed to initialize vector database: {e}")
                raise
        return self.index
    
    def embed_phrases(self, phrases: List[str]) -> np.ndarray:
        """Convert phrases to embeddings"""
        model = self._get_model()
        try:
            # Generate embeddings in batches
            embeddings = model.encode(phrases, batch_size=config.BATCH_SIZE, 
                                      show_progress_bar=False, convert_to_numpy=True)
            logger.info(f"Generated {len(embeddings)} embeddings of shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def add_to_db(self, phrases: List[str]) -> None:
        """Add phrases to the vector database"""
        if not phrases:
            logger.warning("No phrases to add to vector database")
            return
            
        index = self._initialize_db()
        embeddings = self.embed_phrases(phrases)
        
        try:
            # Convert to float32 if not already
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
                
            # Add to index
            index.add(embeddings)
            self.phrases.extend(phrases)
            logger.info(f"Added {len(phrases)} phrases to vector database")
        except Exception as e:
            logger.error(f"Error adding phrases to vector database: {e}")
            raise
    
    def detect_anomalies(self, phrases: List[str], threshold: float = 0.8) -> Dict[str, Any]:
        """Detect anomalies in phrases based on vector similarity"""
        if not self.db_initialized or index.ntotal == 0:
            logger.warning("Vector database not initialized or empty")
            return {"anomalies": [], "scores": []}
            
        if not phrases:
            return {"anomalies": [], "scores": []}
            
        # Get embeddings for input phrases
        query_embeddings = self.embed_phrases(phrases)
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        
        # Search for nearest neighbors
        k = min(10, self.index.ntotal)  # Number of neighbors to consider
        distances, _ = self.index.search(query_embeddings, k)
        
        # Compute anomaly scores (average distance to k nearest neighbors)
        anomaly_scores = np.mean(distances, axis=1)
        
        # Detect anomalies based on threshold
        anomalies = []
        scores = []
        for i, (phrase, score) in enumerate(zip(phrases, anomaly_scores)):
            scores.append(float(score))
            if score > threshold:
                anomalies.append({
                    "phrase": phrase,
                    "score": float(score),
                    "index": i
                })
        
        result = {
            "anomalies": anomalies,
            "scores": scores,
            "avg_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
        }
        
        logger.info(f"Detected {len(anomalies)} anomalies out of {len(phrases)} phrases")
        return result

    def get_phrase_vectors(self, segmented_doc: List[Dict[str, str]]) -> np.ndarray:
        """Extract phrases from segmented document and convert to vectors"""
        phrases = [item['sentence'] for item in segmented_doc]
        return self.embed_phrases(phrases), phrases


# Create a singleton instance
anomaly_detector = PhraseAnomalyDetector(use_gpu=config.USE_CUDA)