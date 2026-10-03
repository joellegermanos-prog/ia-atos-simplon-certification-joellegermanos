"""Pydantic schemas for the Retour-Emploi API — fourni.

Aligned with feature_columns from the Retour-Emploi model metadata.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EmploymentApplication(BaseModel):
    """Employment-return request sent to the CISIA model."""

    usager_id: str | None = Field(default=None, min_length=1, max_length=100)
    session_id: str | None = Field(default=None, min_length=1, max_length=100)
    age: float | None = Field(default=None, ge=16, le=100)
    niveau_diplome: Literal["Sans diplôme", "Bac", "Bac+2", "Bac+5"]
    anciennete_poste_ans: float = Field(..., ge=0)
    code_rome_vise: str
    nationalite_hors_ue: Literal[0, 1] | None = None
    est_allocataire: Literal[0, 1] = 0
    code_insee_commune: str
    synthese_entretien: str


class TrainingRecord(EmploymentApplication):
    classe_retour_emploi: Literal[0, 1, 2]


class TrainRequest(BaseModel):
    records: list[TrainingRecord] = Field(..., min_length=10, max_length=10000)
    experiment_name: str = Field(default="cisia-employment", min_length=1, max_length=100)


class TrainResponse(BaseModel):
    status: Literal["trained"]
    rows: int
    model_version: str
    mlflow_run_id: str | None = None


class Prediction(BaseModel):
    """Multiclass response from the CISIA model."""

    prediction: int = Field(..., ge=0, le=2)
    prediction_label: str
    probabilities: dict[str, float]
    model_version: str
    request_id: str
    usager_id: str | None = None
    session_id: str | None = None
    needs_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)


class Feedback(BaseModel):
    """Validated business label for a prediction already served."""

    request_id: str
    prediction: int = Field(..., ge=0, le=2)
    true_label: int = Field(..., ge=0, le=2)
    comments: str | None = None


class HealthResponse(BaseModel):
    """Output schema for /health."""

    status: str


class InfoResponse(BaseModel):
    """Output schema for /info."""

    api_version: str
    model_name: str
    model_version: str
    model_created_at: str
    metrics_holdout: dict | None = None
    sklearn_version: str | None = None
    dataset_sha256: str | None = None
