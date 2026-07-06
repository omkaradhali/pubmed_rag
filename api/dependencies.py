from fastapi import Depends, Header, HTTPException

from api.config import Settings, get_settings


async def verify_api_key(
    x_api_key: str | None = Header(None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Validate the X-API-Key header when API_KEYS is configured.

    If API_KEYS is empty (the default), auth is disabled — safe for local dev
    and self-hosted deployments that don't need access control. Set API_KEYS
    to one or more comma-separated keys before exposing the API publicly.

    API_KEYS is stored as a raw string (not list[str]) to avoid pydantic-settings
    attempting JSON-decode on the env var before our validator can intercept.
    """
    valid_keys = [k.strip() for k in settings.api_keys.split(",") if k.strip()]
    if not valid_keys:
        return
    if x_api_key is None or x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
