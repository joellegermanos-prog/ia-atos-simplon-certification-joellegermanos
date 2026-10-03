"""Fixtures pytest — service model (fourni).

Ajoute la racine du service au sys.path pour que `from app.main import app`
fonctionne quand pytest est lancé depuis la racine du repo
(`pytest services/model/tests`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT))


@pytest.fixture
def client():
    """TestClient FastAPI (déclenche le lifespan → charge le modèle)."""
    for module_name in ("app.main", "app.middleware", "app.schemas", "app"):
        sys.modules.pop(module_name, None)
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload() -> dict:
    """Une demande d'emploi valide alignée sur LoanApplication."""
    return {
        "niveau_diplome": "Bac+2",
        "anciennete_poste_ans": 3.0,
        "code_rome_vise": "M1805",
        "code_insee_commune": "75056",
        "synthese_entretien": (
            "Recherche un emploi stable dans le domaine informatique."
        ),
    }


