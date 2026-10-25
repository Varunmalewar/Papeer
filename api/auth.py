"""Auth: Neon Auth (Better Auth) JWT preferred, X-API-Key fallback.

- Frontend (jab banega) `Authorization: Bearer <jwt>` bhejega — token
  NEON_AUTH_JWKS_URL se verify hota hai (EdDSA, 15-min expiry).
- Key-based clients / scripts `X-API-Key: <PAPEER_API_KEY>` bhej sakte hai.
- Dono missing → 401. PAPEER_API_KEY unset ho to key-path fail-closed (503),
  JWT path tab bhi chalta hai.
"""
import os
import time
import urllib.request
import json
from urllib.parse import urlsplit

import jwt
from fastapi import Header, HTTPException

_jwks_cache: dict = {"keys": None, "at": 0.0}
JWKS_TTL_S = 600.0


def _issuer() -> str:
    """Neon Auth JWT `iss` claim = base URL ka ORIGIN (scheme + host), path nahi.

    NEON_AUTH_BASE_URL me `/neondb/auth` path hota hai, lekin token ka `iss`
    sirf origin hota hai (e.g. https://xxx.neon.tech). PyJWT exact-match karta
    hai, isliye path hata ke origin hi issuer mana jata hai.
    """
    base = os.environ.get("NEON_AUTH_BASE_URL", "").strip().rstrip("/")
    if not base:
        return ""
    parts = urlsplit(base)
    if not parts.scheme or not parts.netloc:
        return base  # unexpected format — wahi use kar lo
    return f"{parts.scheme}://{parts.netloc}"


def _get_jwks() -> dict:
    jwks_url = os.environ.get("NEON_AUTH_JWKS_URL", "").strip()
    if not jwks_url:
        raise HTTPException(status_code=503, detail="Auth not configured (NEON_AUTH_JWKS_URL).")
    now = time.monotonic()
    if _jwks_cache["keys"] is None or now - _jwks_cache["at"] > JWKS_TTL_S:
        try:
            with urllib.request.urlopen(jwks_url, timeout=10) as resp:
                _jwks_cache["keys"] = json.loads(resp.read().decode())
                _jwks_cache["at"] = now
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Could not fetch JWKS: {e}")
    return _jwks_cache["keys"]


def verify_neon_jwt(token: str) -> dict:
    """Verify a Neon Auth JWT, return its claims. Raises 401 on failure."""
    issuer = _issuer()
    # Also accept full base URL as issuer (some Neon Auth versions include path)
    base = os.environ.get("NEON_AUTH_BASE_URL", "").strip().rstrip("/")
    valid_issuers = [issuer]
    if base and base != issuer:
        valid_issuers.append(base)
    
    try:
        jwks_client = jwt.PyJWKClient(
            os.environ["NEON_AUTH_JWKS_URL"],
            cache_keys=True,
            lifespan=JWKS_TTL_S,
        )
        key = jwks_client.get_signing_key_from_jwt(token).key
        # Try with each valid issuer
        last_error = None
        for valid_issuer in valid_issuers:
            try:
                claims = jwt.decode(
                    token,
                    key,
                    algorithms=["EdDSA"],
                    issuer=valid_issuer or None,
                    audience=None,  # Disable audience check — Neon Auth may include path
                    options={"require": ["exp"]},
                    leeway=30,
                )
                print(f"[auth] JWT verified: sub={claims.get('sub')}, iss={claims.get('iss')}, aud={claims.get('aud')}, exp={claims.get('exp')}")
                return claims
            except Exception as e:
                last_error = e
                print(f"[auth] JWT verify failed for issuer={valid_issuer}: {e}")
                continue
        # If all issuer options failed
        raise last_error or Exception("No valid issuer matched")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[auth] JWT verify failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def require_auth(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> dict:
    """Accept Bearer JWT or API key. Returns identity dict for the request."""
    if authorization and authorization.lower().startswith("bearer "):
        claims = verify_neon_jwt(authorization.split(None, 1)[1])
        return {"method": "jwt", "sub": claims.get("sub"), "claims": claims}
    expected = os.environ.get("PAPEER_API_KEY", "").strip()
    if x_api_key and expected and x_api_key == expected:
        return {"method": "api_key", "sub": "api-key"}
    if x_api_key and not expected:
        raise HTTPException(status_code=503, detail="API key not configured (PAPEER_API_KEY).")
    raise HTTPException(status_code=401, detail="Missing credentials (Bearer JWT or X-API-Key).")


# Backwards-compat alias: existing routers import require_api_key.
def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> dict:
    return require_auth(authorization, x_api_key)


def require_user(x_papeer_user: str | None = Header(default=None)) -> str:
    """Trusted per-user scope id (server-derived library id), forwarded by the
    Next.js proxy ONLY after verifying the browser's Neon Auth session.

    The client can never set this meaningfully — every vector read/write is
    additionally filtered by this value, so client-supplied session/scope ids
    cannot touch another user's documents."""
    value = (x_papeer_user or "").strip()
    if not value:
        raise HTTPException(status_code=401, detail="User identity required.")
    return value

