import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from typing import cast, List
import logging
from pathlib import Path

from .config import settings
from .embeddings_abc import Embedder

logger = logging.getLogger("embedding-api")


class E5_Embedder(Embedder):
    """
    E5 embedder class using SentenceTransformers.
    """
    
    def __init__(self):
        self._model: SentenceTransformer | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._dim: int = 0
        self._load_model()
    
    
    @property
    def dim(self) -> int:
        """
        Return the dimension of the embeddings.
        """
        
        return self._dim
    
    
    def _load_model(self) -> None:
        """
        Load the E5 model and tokenizer.
        """
        
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        # Determine model source (local / HF)
        if settings.USE_LOCAL_MODEL:
            local_path = Path(settings.BASE_DIR) / settings.MODEL_PATH
            source = str(local_path)
            local_files_only = True
            logger.info(f"Using local model: {source}")
        else:
            source = settings.MODEL_ID
            local_files_only = False
            logger.info(f"Using Hugging Face model: {source}")
        
        logger.info(f"Loading model on device: {device}")
        
        try:
            self._model = SentenceTransformer(source, device=device, local_files_only=local_files_only)
            self._model.eval()
            self._tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(source, local_files_only=local_files_only))
            
            # Determine embedding dimension
            with torch.inference_mode():
                test_passage = self._model.encode(["passage: test"], normalize_embeddings=True, convert_to_numpy=True)
            self._dim = int(test_passage.shape[1])
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dim}")
        
        except Exception as e:
            logger.error(f"Failed to load model from {source}: {e}")
            
            if local_files_only:
                logger.info("Falling back to Hugging Face model...")
                
                try:
                    self._model = SentenceTransformer(settings.MODEL_ID, device=device, local_files_only=False)
                    self._model.eval()
                    self._tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(settings.MODEL_ID, local_files_only=False))
                    
                    with torch.inference_mode():
                        probe = self._model.encode(["passage: probe"], normalize_embeddings=True, convert_to_numpy=True)
                    self._dim = int(probe.shape[1])
                    
                    logger.info(f"Fallback model loaded successfully. Embedding dimension: {self.dim}")
                except Exception as fallback_error:
                    logger.error(f"Fallback to Hugging Face also failed: {fallback_error}")
                    raise fallback_error
            else:
                raise e


    def encode(self, texts: List[str], input_type: str, normalize: bool, batch_size: int) -> np.ndarray:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: List of input texts
            input_type: Type of input ("query" or "passage")
            normalize: Whether to normalize embeddings
            batch_size: Batch size for processing
        """
        
        assert self._model is not None, "Model must be loaded"
        
        prefixed_texts = [f"{input_type}: {text}" for text in texts]
        
        with torch.inference_mode():
            embeddings = self._model.encode(
                prefixed_texts,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        
        return embeddings


    def tokenize_count(self, texts: List[str], input_type: str) -> int:
        """
        Count tokens for the given texts including E5 prefixes.

        Args:
            texts: List of input texts
            input_type: Type of input ("query" or "passage")
        """
        
        assert self._tokenizer is not None, "Tokenizer must be loaded"
        
        prefixed_texts = [f"{input_type}: {text}" for text in texts]
        
        enc: BatchEncoding = self._tokenizer(
            prefixed_texts, 
            padding=False, 
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )  # type: ignore[assignment]
        
        return sum(len(ids) for ids in enc["input_ids"])  # type: ignore[arg-type]