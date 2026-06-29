import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.config import get_settings
from api.limiter import limiter
from api.logging_config import configure_logging, request_id_var
from api.routers import ask, cds_hooks, health
from pubmed_rag import __version__

logger = logging.getLogger(__name__)

_settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging and emit startup/shutdown log events."""
    configure_logging(_settings.log_level)
    logger.info("pubmed_rag API starting", extra={"llm_provider": _settings.llm_provider})
    yield
    logger.info("pubmed_rag API stopped")


app = FastAPI(title="pubmed_rag", version=__version__, lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def inject_request_id(request: Request, call_next):
    """Attach a request UUID, inject it into all log records, and return it as X-Request-ID."""
    rid = str(uuid.uuid4())
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        request_id_var.reset(token)


app.include_router(health.router)
app.include_router(ask.router)
app.include_router(cds_hooks.router)
