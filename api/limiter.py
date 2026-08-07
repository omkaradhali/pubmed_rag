"""
limiter.py — Shared slowapi rate limiter keyed by client IP address.

Defined in its own module so routers and the app factory import the same
Limiter instance without creating an import cycle through api.main.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Keyed on the remote address so limits are enforced per client IP.
limiter = Limiter(key_func=get_remote_address)
