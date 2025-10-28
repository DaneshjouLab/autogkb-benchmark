"""
Model caching using singleton pattern for PubMedBERT to avoid reloading.
"""

import threading
from typing import Optional
from sentence_transformers import SentenceTransformer


class ModelCache:
    """
    Singleton class for caching the PubMedBERT model.
    Thread-safe implementation with lazy loading.
    """
    
    _instance: Optional['ModelCache'] = None
    _lock = threading.Lock()
    _model: Optional[SentenceTransformer] = None
    
    def __new__(cls) -> 'ModelCache':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """
        Get the cached PubMedBERT model, loading it if necessary.
        
        Returns:
            SentenceTransformer: The PubMedBERT model instance
        """
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    print("Loading PubMedBERT model...")
                    cls._model = SentenceTransformer(
                        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
                    )
                    print("PubMedBERT model loaded successfully.")
        return cls._model
    
    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the cached model (useful for testing or memory management).
        """
        with cls._lock:
            cls._model = None
