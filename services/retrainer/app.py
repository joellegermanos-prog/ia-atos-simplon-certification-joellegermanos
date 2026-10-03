from __future__ import annotations

import asyncio
import hmac
import json
import os
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, status
from loguru import logger

RETRAIN_RESULT_PATH = Path(
    os.environ.get("RETRAIN_RESULT_PATH", "/app/reports/retrain-result.json")
)

app = FastAPI(title="Retour-Emploi Feedback Retrainer", version="1.0.0")
app.state.retrain_task = None
app.state.retrain_job = None


def _latest_result() -> dict[str, object]:
    if not RETRAIN_RESULT_PATH.is_file():
        return {}
    try:
        result = json.loads(RETRAIN_RESULT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        key: result[key]
        for key in ("status", "new_feedbacks", "min_feedback", "promoted_model_version")
        if key in result
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/retrain", status_code=status.HTTP_202_ACCEPTED)
async def trigger_retraining(
    x_train_token: str | None = Header(default=None),
) -> dict[str, object]:
    expected_token = os.environ.get("RETRAIN_API_TOKEN", "")
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retraining trigger is not configured",
        )
    if x_train_token is None or not hmac.compare_digest(x_train_token, expected_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Training forbidden")

    current_task = app.state.retrain_task
    if current_task is not None and not current_task.done():
        return {"job_id": app.state.retrain_job["job_id"], "status": "already_running"}

    job_id = str(uuid.uuid4())
    app.state.retrain_job = {"job_id": job_id, "status": "running"}
    app.state.retrain_task = asyncio.create_task(_run_retraining(job_id))
    return {"job_id": job_id, "status": "started"}


@app.get("/retrain/status")
async def retraining_status() -> dict[str, object]:
    if app.state.retrain_job is not None:
        return app.state.retrain_job
    return {"status": "idle", "last_run": _latest_result()}


async def _run_retraining(job_id: str) -> None:
    min_feedback = max(1, int(os.environ.get("RETRAIN_MIN_FEEDBACK", "200")))
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "scripts/retrain.py",
            "--min-feedback",
            str(min_feedback),
            cwd="/app",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        await process.communicate()
        result = _latest_result()
        logger.info(
            "Retraining job {job_id} exited with code {exit_code} and status {result_status}",
            job_id=job_id,
            exit_code=process.returncode,
            result_status=result.get("status", "unknown"),
        )
        app.state.retrain_job = {
            "job_id": job_id,
            "status": "completed" if process.returncode == 0 else "failed",
            "exit_code": process.returncode,
            "result": result,
        }
    except (OSError, RuntimeError, ValueError, asyncio.SubprocessError) as exc:
        logger.exception("Retraining job {job_id} failed", job_id=job_id)
        app.state.retrain_job = {
            "job_id": job_id,
            "status": "failed",
            "error": type(exc).__name__,
            "result": _latest_result(),
        }