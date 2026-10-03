"""Fixtures pytest — service backend.

Ajoute la racine du service au sys.path pour que `from app.main import app`
fonctionne quand pytest est lancé depuis la racine du repo
(`pytest services/backend/tests`).
"""
from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT))

# The model and backend test suites both expose a local package named `app`.
# Remove a previously collected model package so global pytest collection
# resolves this suite against services/backend/app.
for module_name in ("app.main", "app.middleware", "app.schemas", "app"):
	sys.modules.pop(module_name, None)
