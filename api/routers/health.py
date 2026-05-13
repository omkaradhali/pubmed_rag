from fastapi import APIRouter

from api.schemas import HealthResponse
from pubmed_rag import __version__

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
)
async def health_check() -> HealthResponse:
    """
    Returns 200 when the API is running.

    Use this endpoint to verify the server is up before sending queries.
    No authentication required.
    """
    return HealthResponse(status="ok", version=__version__)
