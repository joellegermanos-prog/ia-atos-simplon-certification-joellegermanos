import importlib.util
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

MODULE_PATH = Path(__file__).parents[1] / "app.py"
SPEC = importlib.util.spec_from_file_location("retrainer_api", MODULE_PATH)
retrainer_api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retrainer_api)


def test_retrain_requires_configured_token(monkeypatch):
    monkeypatch.delenv("RETRAIN_API_TOKEN", raising=False)
    response = TestClient(retrainer_api.app).post("/retrain")

    assert response.status_code == 503


def test_retrain_rejects_invalid_token(monkeypatch):
    monkeypatch.setenv("RETRAIN_API_TOKEN", "secret")
    response = TestClient(retrainer_api.app).post(
        "/retrain",
        headers={"X-Train-Token": "wrong"},
    )

    assert response.status_code == 403


def test_retrain_starts_one_background_job(monkeypatch):
    monkeypatch.setenv("RETRAIN_API_TOKEN", "secret")
    retrainer_api.app.state.retrain_task = None
    retrainer_api.app.state.retrain_job = None
    scheduled = []

    def fake_create_task(coroutine):
        coroutine.close()
        task = SimpleNamespace(done=lambda: False)
        scheduled.append(task)
        return task

    monkeypatch.setattr(
        retrainer_api,
        "asyncio",
        SimpleNamespace(create_task=fake_create_task),
    )
    client = TestClient(retrainer_api.app)
    headers = {"X-Train-Token": "secret"}

    first = client.post("/retrain", headers=headers)
    second = client.post("/retrain", headers=headers)

    assert first.status_code == 202
    assert first.json()["status"] == "started"
    assert second.status_code == 202
    assert second.json()["status"] == "already_running"
    assert len(scheduled) == 1