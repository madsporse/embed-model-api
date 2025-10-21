from enum import Enum
from typing import List, Union, Optional
from pydantic import BaseModel, Field, field_validator


class InputType(str, Enum):
    query = "query"
    passage = "passage"


class EmbedRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Single string or list of strings to embed")
    input_type: InputType = Field(InputType.passage, description="E5-prefix")
    normalize: bool = Field(True, description="L2 normalize embeddings")
    batch_size: Optional[int] = Field(None, ge=1, le=512, description="Override default batch size")
    truncate: Optional[int] = Field(None, ge=32, le=8192, description="Max tokens/characters per item (best-effort)")

    @field_validator("input")
    @classmethod
    def non_empty(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("input cannot be empty string")
        if isinstance(v, list) and not v:
            raise ValueError("input list cannot be empty")
        return v


class EmbeddingItem(BaseModel):
    index: int
    embedding: List[float]


class Usage(BaseModel):
    total_input_tokens: int


class EmbedResponse(BaseModel):
    model: str
    data: List[EmbeddingItem]
    embedding_dimension: int
    usage: Usage


class ErrorResponse(BaseModel):
    detail: str
