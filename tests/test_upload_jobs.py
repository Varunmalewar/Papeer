"""Uploads return 202 at once and ingest in the background (ingestion is mocked)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
import api.routers.papers as papers

PDF = b"%PDF-1.4 minimal"


@pytest.fixture
def client(monkeypatch):
    # patch first: importing backend.vector_store runs load_dotenv(override=True),
    # which would otherwise replace the test key set below
    monkeypatch.setattr("backend.vector_store.count_chunks", lambda *a, **k: 7)
    monkeypatch.setenv("PAPEER_API_KEY", "test-key")
    monkeypatch.setattr(papers, "_JOBS", {})
    return TestClient(app)


def _headers(user="lib-a"):
    return {"X-API-Key": "test-key", "X-Papeer-User": user}


def _upload(client, user="lib-a", name="p.pdf", data=PDF):
    return client.post("/papers/upload?session_id=s", headers=_headers(user), files={"file": (name, data)})


def test_upload_returns_202_then_job_is_done(client, monkeypatch):
    monkeypatch.setattr(papers, "ingest_pdf_bytes", lambda *a, **k: ["doc"])
    r = _upload(client)
    assert r.status_code == 202 and r.json()["status"] == "processing"
    job = client.get(f"/papers/jobs/{r.json()['job_id']}", headers=_headers()).json()
    assert job == {"state": "done", "chunks": 7, "error": None}


def test_ingestion_failure_is_reported_by_the_job(client, monkeypatch):
    def boom(*a, **k):
        raise ValueError("bad pdf")

    monkeypatch.setattr(papers, "ingest_pdf_bytes", boom)
    job_id = _upload(client).json()["job_id"]
    job = client.get(f"/papers/jobs/{job_id}", headers=_headers()).json()
    assert job["state"] == "error" and "bad pdf" in job["error"]


def test_empty_result_is_an_error_job(client, monkeypatch):
    monkeypatch.setattr(papers, "ingest_pdf_bytes", lambda *a, **k: [])
    job_id = _upload(client).json()["job_id"]
    job = client.get(f"/papers/jobs/{job_id}", headers=_headers()).json()
    assert job["state"] == "error" and "No readable text" in job["error"]


def test_other_user_cannot_read_a_job(client, monkeypatch):
    monkeypatch.setattr(papers, "ingest_pdf_bytes", lambda *a, **k: ["doc"])
    job_id = _upload(client, user="lib-a").json()["job_id"]
    assert client.get(f"/papers/jobs/{job_id}", headers=_headers("lib-b")).status_code == 404


def test_bad_file_is_rejected_before_any_job(client):
    assert _upload(client, name="p.pdf", data=b"not a pdf").status_code == 400
    assert _upload(client, name="p.exe").status_code == 400
    assert papers._JOBS == {}
