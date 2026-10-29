"""Who may read which library, the per-user rate limit, and the server-side URL fetch guard."""
import hashlib

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

import api.rate_limit as rl
from api import auth
from api.main import app
import backend.paper_loader as pl


# ── URL fetch guard ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/admin",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/",
    "http://[::1]/",
    "ftp://example.com/file",
    "file:///etc/passwd",
])
def test_non_public_or_non_http_urls_are_refused(url):
    with pytest.raises(ValueError):
        pl._assert_public_url(url)


def test_public_address_is_allowed():
    pl._assert_public_url("http://93.184.216.34/")  # an IP literal, so no DNS lookup


def test_host_that_resolves_to_a_private_address_is_refused(monkeypatch):
    monkeypatch.setattr(pl.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("10.1.2.3", 80))])
    with pytest.raises(ValueError):
        pl._assert_public_url("http://looks-public.example/")


def test_webpage_endpoint_answers_400_for_a_private_url(monkeypatch):
    # patch first: importing backend.vector_store runs load_dotenv(override=True)
    monkeypatch.setattr("backend.vector_store.count_chunks", lambda *a, **k: 0)
    monkeypatch.setenv("PAPEER_API_KEY", "test-key")
    r = TestClient(app).post(
        "/papers/webpage?session_id=s&url=http://127.0.0.1:9/x",
        headers={"X-API-Key": "test-key", "X-Papeer-User": "lib-a"},
    )
    assert r.status_code == 400


# ── user scope ───────────────────────────────────────────────────────────────

def test_login_token_cannot_choose_another_users_library():
    ident = {"method": "jwt", "claims": {"email": "Alice@Example.com"}}
    got = auth.require_user(ident=ident, x_papeer_user="lib-victim")
    assert got == "lib-" + hashlib.sha256(b"alice@example.com").hexdigest()[:32]


def test_login_token_without_email_is_refused():
    with pytest.raises(HTTPException) as e:
        auth.require_user(ident={"method": "jwt", "claims": {}}, x_papeer_user="lib-x")
    assert e.value.status_code == 403


def test_proxy_with_api_key_sets_the_library_through_the_header():
    assert auth.require_user(ident={"method": "api_key"}, x_papeer_user=" lib-a ") == "lib-a"
    with pytest.raises(HTTPException) as e:
        auth.require_user(ident={"method": "api_key"}, x_papeer_user=None)
    assert e.value.status_code == 401


# ── rate limit ───────────────────────────────────────────────────────────────

def _req(user):
    return Request({"type": "http", "client": ("1.2.3.4", 1),
                    "headers": [(b"x-api-key", b"shared-proxy-key"), (b"x-papeer-user", user.encode())]})


def test_one_busy_user_does_not_block_another(monkeypatch):
    monkeypatch.setattr(rl, "_hits", {})
    for _ in range(20):
        rl.rate_limit(_req("lib-a"))
    with pytest.raises(HTTPException):
        rl.rate_limit(_req("lib-a"))  # user A is limited ...
    rl.rate_limit(_req("lib-b"))      # ... user B is not
