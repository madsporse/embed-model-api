from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Embedder(ABC):
    """Abstract base class defining the interface for embedding models."""
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimension of the embeddings."""
        ...
    
    @abstractmethod
    def encode(self, texts: List[str], input_type: str, normalize: bool, batch_size: int) -> np.ndarray:
        """Generate embeddings for the given texts."""
        ...
    
    @abstractmethod
    def tokenize_count(self, texts: List[str], input_type: str) -> int:
        """Count tokens for the given texts."""
        ...