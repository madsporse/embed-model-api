import time, uuid, logging
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import settings
from .schemas import EmbedRequest, EmbedResponse, EmbeddingItem, ErrorResponse, Usage
from .embeddings_abc import Embedder
from .embeddings import E5_Embedder


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger("embedding-api")

_embedder: Embedder | None = None  # global embedder instance for dependency injection


def get_embedder() -> Embedder:
    """Dependency function to provide the embedder instance."""
    assert _embedder is not None, "Embedder not initialized"
    return _embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application startup and shutdown.
    
    Handles model loading on startup and potential cleanup on shutdown.
    """
    global _embedder
    logger.info("Starting up embedding API...")
    _embedder = E5_Embedder()
    logger.info("Startup complete")
    
    yield  # app runs here
    
    logger.info("Shutting down embedding API...")

tags_metadata = [
    {"name": "embeddings", "description": "E5 embedding endpoint"},
    {"name": "health", "description": "Health & readiness checks"},
]


app = FastAPI(
    title="Embedding API (multilingual-e5-large)",
    version="1.0.0",
    description="Produktionsklart API til E5 embeddings (FastAPI).",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# CORS setup
if settings.CORS_ALLOW_ALL:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """
    Adds request ID to headers and log request details with timing.
    """
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    
    try:
        response = await call_next(request)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        client_host = request.client.host if request.client else "unknown"
        logger.info(
            f"%s %s %s %0.2fms rid=%s",
            request.method, 
            request.url.path, 
            client_host, 
            elapsed_ms, 
            req_id
        )
        
    response.headers["X-Request-ID"] = req_id
    
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    """
    Global HTTP exception handler.
    
    Returns JSON error responses with status code and detail message.
    """
    
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/healthz", tags=["health"])
def healthz():
    """
    Basic health check endpoint.
    """
    
    return {"status": "ok"}


@app.get("/readyz", tags=["health"])
def readyz():
    """
    Basic readiness endpoint.
    """
    
    return {"status": "ready"}


@app.post("/embed", response_model=EmbedResponse, responses={400: {"model": ErrorResponse}}, tags=["embeddings"])
async def embed(req: EmbedRequest, embedder: Embedder = Depends(get_embedder)):
    """
    Generate embeddings for input text(s) using embedding model.
    
    Validates input size and character limits, applies appropriate prefixes,
    and returns normalized embeddings with usage statistics.
    
    Args:
        req: Embedding request containing text(s), input type, and options
        embedder: Injected embedder instance
        
    Returns:
        EmbedResponse with embeddings, model info, and token usage
    """
    
    # Normalize input to list
    texts = [req.input] if isinstance(req.input, str) else list(req.input)

    # Input validation
    if len(texts) > settings.MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"Max batch size: {settings.MAX_BATCH}")
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise HTTPException(status_code=400, detail=f"Item {i} is not a string")
        if len(t) > settings.MAX_CHARS_PER_ITEM:
            raise HTTPException(status_code=413, detail=f"Item {i} exceeds {settings.MAX_CHARS_PER_ITEM} characters")

    batch_size = req.batch_size or settings.DEFAULT_BATCH_SIZE

    # Generate embeddings and count tokens
    try:
        vecs = await run_in_threadpool(
            embedder.encode, texts, req.input_type.value, req.normalize, batch_size
        )
        
        total_tokens = await run_in_threadpool(
            embedder.tokenize_count, texts, req.input_type.value
        )
    except Exception as e:
        logger.exception("Infer error")
        raise HTTPException(status_code=500, detail=str(e))

    data = [EmbeddingItem(index=i, embedding=vec.tolist()) for i, vec in enumerate(vecs)]
    
    return EmbedResponse(
        model="intfloat/multilingual-e5-large",
        data=data,
        embedding_dimension=embedder.dim,
        usage=Usage(total_input_tokens=int(total_tokens)),
    )
