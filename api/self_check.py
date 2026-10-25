"""Self-check for api/auth.py + api/rate_limit.py (stdlib only, no pytest needed)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["PAPEER_API_KEY"] = "test-key"
os.environ["PAPEER_RATE_LIMIT_PER_MIN"] = "3"

from fastapi import HTTPException  # noqa: E402

from api.auth import require_api_key  # noqa: E402
from api.rate_limit import _hits, rate_limit  # noqa: E402


class _Req:
    def __init__(self, key=None, host="1.2.3.4"):
        self.headers = {"X-API-Key": key} if key else {}
        self.client = type("C", (), {"host": host})()


# auth: correct key passes
require_api_key(None, "test-key")

# auth: wrong / missing creds → 401
for bad in ("wrong", None):
    try:
        require_api_key(None, bad)
        raise AssertionError(f"auth accepted {bad!r}")
    except HTTPException as e:
        assert e.status_code == 401, e.status_code

# auth: garbage Bearer → 401 (JWKS fetch would fail, no network in test)
os.environ["NEON_AUTH_JWKS_URL"] = "https://example.invalid/.well-known/jwks.json"
os.environ["NEON_AUTH_BASE_URL"] = "https://example.invalid/auth"
try:
    require_api_key("Bearer garbage.token.here", None)
    raise AssertionError("auth accepted garbage JWT")
except HTTPException as e:
    assert e.status_code in (401, 503), e.status_code

# auth: key not configured → key-path fail closed (503), JWT path unaffected
del os.environ["PAPEER_API_KEY"]
try:
    require_api_key(None, "test-key")
    raise AssertionError("auth passed with no key configured")
except HTTPException as e:
    assert e.status_code == 503, e.status_code
os.environ["PAPEER_API_KEY"] = "test-key"

# rate limit: 3 pass, 4th → 429
_hits.clear()
req = _Req(key="k")
for _ in range(3):
    rate_limit(req)
try:
    rate_limit(req)
    raise AssertionError("rate limit did not trip")
except HTTPException as e:
    assert e.status_code == 429, e.status_code

print("api self-check OK")
