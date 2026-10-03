from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train as train_module
from evaluate import (
    evaluate_model,
    load_text_dataset,
    rebuild_holdout,
    validate_dataset,
)
from evaluation_utils import print_metrics
from train import (
    _critical_error_0_to_2,
    _select_best_grid_search_candidate,
    _summarize_grid_search_results,
    build_fit_parameters,
    build_robustness_table,
    compare_baseline_hyperparameters,
    compute_stratified_cv_summary,
    measure_training_inference_cpu,
    print_training_summary,
)


@pytest.mark.parametrize(
    ("model_type", "config_name", "expected_class_2_weight"),
    [
        ("xgboost", "best_class_2_ethique", 2.5),
        ("lightgbm", "best_class_2_complet", 2.5),
        ("xgboost", "custom_class_2_heavy", 4.0),
    ],
)
def test_build_fit_parameters_weights_class_2_named_config(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    config_name: str,
    expected_class_2_weight: float,
) -> None:
    if config_name == "custom_class_2_heavy":
        monkeypatch.setitem(
            train_module.MODEL_CONFIGS[model_type],
            config_name,
            {
                "class_weight": {
                    0: 1.0,
                    1: 1.0,
                    2: expected_class_2_weight,
                },
            },
        )

    parameters = build_fit_parameters(
        model_type=model_type,
        config_name=config_name,
        y_train=pd.Series([0, 1, 2, 2]),
    )

    np.testing.assert_array_equal(
        parameters["classifier__sample_weight"],
        [1.0, 1.0, expected_class_2_weight, expected_class_2_weight],
    )


def test_grid_search_summary_reports_positive_error_rate() -> None:
    grid_search = SimpleNamespace(
        cv_results_={
            "params": [{"candidate": "lower"}, {"candidate": "higher"}],
            "mean_test_f1_macro": [0.7, 0.6],
            "mean_test_f1_class_2": [0.4, 0.7],
            "mean_test_recall_class_2": [0.5, 0.8],
            "mean_test_error_2_to_0": [-0.25, -0.1],
            "mean_test_error_0_to_2": [-0.05, -0.2],
            "rank_test_f1_macro": [1, 2],
            "rank_test_f1_class_2": [2, 1],
        }
    )

    summary = _summarize_grid_search_results(grid_search)

    assert summary["params"].tolist() == [
        {"candidate": "higher"},
        {"candidate": "lower"},
    ]
    assert summary["mean_test_error_2_to_0"].tolist() == [0.1, 0.25]
    assert summary["mean_test_error_0_to_2"].tolist() == [0.2, 0.05]
    assert summary["mean_test_combined_error_cost"].tolist() == [0.5, 0.8]


def test_grid_search_refit_minimizes_weighted_critical_errors_first() -> None:
    cv_results = {
        "mean_test_f1_macro": [0.99, 0.60, 0.70, 0.90],
        "mean_test_error_2_to_0": [-0.10, 0.0, -0.05, -0.20],
        "mean_test_error_0_to_2": [0.0, -0.25, -0.10, -0.25],
    }

    assert _select_best_grid_search_candidate(cv_results) == 2


def test_critical_error_0_to_2_returns_rate_among_true_class_zero() -> None:
    y_true = pd.Series([0, 0, 1, 2])
    y_pred = np.array([2, 1, 2, 0])

    assert _critical_error_0_to_2(y_true, y_pred) == 0.5


def test_default_grid_search_candidate_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_counts: list[int] = []
    scorer_names: list[set[str]] = []
    fit_parameters_seen: list[dict[str, object]] = []

    class DummyPipeline:
        def set_params(self, **parameters: object) -> DummyPipeline:
            return self

    class GridSearchSpy:
        def __init__(self, **kwargs: object) -> None:
            candidate_counts.append(
                len(ParameterGrid(kwargs["param_grid"]))
            )
            scorer_names.append(set(kwargs["scoring"]))
            self.cv_results_ = {
                "params": [{}],
                "mean_test_f1_macro": [0.5],
                "mean_test_f1_class_2": [0.5],
                "mean_test_recall_class_2": [0.5],
                "mean_test_error_2_to_0": [-0.1],
                "mean_test_error_0_to_2": [-0.05],
                "rank_test_f1_macro": [1],
                "rank_test_f1_class_2": [1],
            }

        def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            **fit_parameters: object,
        ) -> GridSearchSpy:
            fit_parameters_seen.append(fit_parameters)
            return self

    monkeypatch.setattr(train_module, "GridSearchCV", GridSearchSpy)
    monkeypatch.setattr(
        train_module,
        "build_training_pipeline",
        lambda **kwargs: DummyPipeline(),
    )

    X_train = pd.DataFrame({"feature": range(6)})
    y_train = pd.Series([0, 0, 1, 1, 2, 2])
    searches = (
        lambda: train_module.grid_search_random_forest_class_2(
            X_train, y_train, scenario_name="texte_seul", cv_folds=2
        ),
        lambda: train_module.grid_search_logistic_regression_class_2(
            "texte_seul", X_train, y_train, cv_folds=2
        ),
        lambda: train_module.grid_search_lightgbm_class_2(
            "texte_seul", X_train, y_train, cv_folds=2
        ),
        lambda: train_module.grid_search_xgboost_class_2(
            "texte_seul", X_train, y_train, cv_folds=2
        ),
    )

    for search in searches:
        search()

    assert candidate_counts == [32, 8, 64, 16]
    assert all("error_0_to_2" in names for names in scorer_names)
    np.testing.assert_array_equal(
        fit_parameters_seen[-1]["classifier__sample_weight"],
        [1.0, 1.0, 1.0, 1.0, 2.5, 2.5],
    )


def test_validate_dataset_accepts_matching_hash(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.csv"
    data_path.write_text("a,b\n1,2\n", encoding="utf-8")
    expected_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    metadata = {"dataset": {"sha256": expected_hash}}

    validate_dataset(data_path, metadata)


def test_validate_dataset_rejects_modified_dataset(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.csv"
    data_path.write_text("a,b\n1,2\n", encoding="utf-8")

    metadata = {"dataset": {"sha256": "wrong_hash"}}

    with pytest.raises(ValueError, match="Le dataset a changé"):
        validate_dataset(data_path, metadata)


def test_load_text_dataset_returns_single_text_column(tmp_path: Path) -> None:
    data_path = tmp_path / "text.csv"
    pd.DataFrame({
        "synthese_entretien": ["Bonjour", "Merci"],
        "classe_retour_emploi": [0, 1],
    }).to_csv(data_path, index=False)

    X, y = load_text_dataset(data_path)

    assert list(X.columns) == ["synthese_entretien"]
    assert X.shape[1] == 1
    assert len(y) == len(X)
    assert set(y.unique()) <= {0, 1}


def test_rebuild_holdout_uses_test_indices(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "synthese_entretien": ["a", "b", "c", "d"],
        "classe_retour_emploi": [0, 1, 2, 0],
    })
    df.to_csv(data_path, index=False)

    metadata = {
        "scenario": {"text_only": True, "include_sensitive": False, "include_text": True},
        "dataset": {"test_indices": [1, 3], "test_rows": 2},
    }

    X_holdout, y_holdout = rebuild_holdout(data_path, metadata)

    assert list(X_holdout.columns) == ["synthese_entretien"]
    assert len(X_holdout) == 2
    assert list(y_holdout.index) == [1, 3]


def test_evaluate_model_returns_metrics_and_predictions(capsys) -> None:
    class DummyModel:
        def predict(self, X):
            return np.array([0, 1, 2, 0])

        def predict_proba(self, X):
            return np.array([
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.8, 0.1, 0.1],
            ])

    y_holdout = pd.Series([0, 1, 2, 0], name="classe_retour_emploi")
    X_holdout = pd.DataFrame({"synthese_entretien": ["a", "b", "c", "d"]})

    metrics, y_pred, y_proba, report_text = evaluate_model(DummyModel(), X_holdout, y_holdout)

    assert set(metrics.keys()) >= {
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "f1_class_0",
        "f1_class_1",
        "f1_class_2",
        "confusion_matrix",
        "classification_report",
        "roc_auc_ovr_macro",
        "cpu_ms_1000_predictions",
    }
    assert metrics["cpu_ms_1000_predictions"] == pytest.approx(
        metrics["cpu_mean_ms"] * 1_000
    )
    assert np.array_equal(y_pred, np.array([0, 1, 2, 0]))
    assert y_proba is not None
    assert "precision" in report_text

    print_metrics(metrics, report_text)
    cpu_output_line = next(
        line for line in capsys.readouterr().out.splitlines()
        if "Temps CPU / 1 000 prédictions" in line
    )
    cpu_milliseconds = float(
        cpu_output_line.split(": ", maxsplit=1)[1].removesuffix(" ms")
    )
    assert cpu_milliseconds == pytest.approx(
        metrics["cpu_ms_1000_predictions"]
    )


def test_measure_training_inference_cpu_reports_per_1000_predictions() -> None:
    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    X_train = pd.DataFrame({"feature": [1, 2, 3]})

    metrics = measure_training_inference_cpu(DummyModel(), X_train)

    assert metrics["cpu_ms_1000_predictions"] >= 0
    assert metrics["inference_samples"] == len(X_train)


def test_training_summary_omits_cpu_predictions(capsys) -> None:
    print_training_summary(
        [
            {
                "scenario_name": "texte_seul",
                "model_type": "random_forest",
                "config_name": "balanced",
                "training_seconds": 1.0,
                "cpu_ms_1000_predictions": 123.4,
                "cv_summary": {"metrics": {}},
            }
        ]
    )

    output = capsys.readouterr().out
    assert "CPU ms / 1k préd. (train)" not in output
    assert "123.400" not in output


def test_compute_stratified_cv_summary_returns_metrics() -> None:
    documents = [
        "je cherche un emploi stable et durable",
        "je veux travailler dans une entreprise",
        "je souhaite un poste administratif",
        "je recherche un emploi en informatique",
        "je veux une opportunite de travail",
        "je cherche un emploi qualifie",
        "je veux developper mon experience professionnelle",
        "je recherche un poste technique",
        "je veux travailler dans le commerce",
        "je cherche un travail sur la finance",
        "je suis disponible pour un emploi",
        "je cherche un poste dans le marketing",
        "je veux une opportunite pour la logistique",
        "je recherche un travail en gestion",
        "je cherche emploi en service client",
        "je souhaite un contrat stable",
        "je veux travailler dans la restauration",
        "je cherche un poste dans la production",
        "je veux un emploi dans la vente",
        "je cherche un travail de bureau",
        "je souhaite un emploi en conception",
        "je cherche du travail pour un projet",
        "je veux un poste dans l'innovation",
        "je recherche un emploi de consultant",
        "je cherche une mission courte",
        "je veux travailler dans la communication",
        "je cherche un poste dans le digital",
        "je veux un emploi dans la data",
        "je recherche un travail de gestionnaire",
        "je cherche un poste en ressources humaines",
    ]

    X = pd.DataFrame({"synthese_entretien": documents})
    y = pd.Series([0, 1, 2] * 10, name="classe_retour_emploi")

    result = compute_stratified_cv_summary(
        model_type="logistic_regression",
        scenario_name="texte_seul",
        config_name="balanced",
        X=X,
        y=y,
        n_splits=3,
    )

    assert result["enabled"] is True
    assert result["n_splits"] == 3
    assert set(result["metrics"].keys()) >= {"accuracy", "f1_macro", "f1_weighted"}
    assert 0.0 <= result["metrics"]["accuracy"]["mean"] <= 1.0


def test_build_robustness_table_returns_std_by_scenario() -> None:
    results = [
        {
            "model_type": "random_forest",
            "scenario_name": "texte_seul",
            "config_name": "balanced",
            "cv_summary": {
                "metrics": {
                    "f1_macro": {"mean": 0.70, "std": 0.08},
                },
            },
        },
        {
            "model_type": "logistic_regression",
            "scenario_name": "texte_seul",
            "config_name": "balanced",
            "cv_summary": {
                "metrics": {
                    "f1_macro": {"mean": 0.75, "std": 0.03},
                },
            },
        },
    ]

    table = build_robustness_table(results)

    assert list(table.columns) == [
        "scenario_name",
        "model_type",
        "config_name",
        "metric",
        "mean",
        "std",
    ]
    assert list(table["model_type"]) == [
        "logistic_regression",
        "random_forest",
    ]
    assert list(table["std"]) == [0.03, 0.08]


def test_compare_baseline_hyperparameters_selects_one_configuration() -> None:
    documents = [
        "emploi stable administratif",
        "travail informatique entreprise",
        "poste commerce vente",
        "emploi gestion projet",
        "travail restauration service",
        "poste finance data",
    ] * 5
    X = pd.DataFrame({"synthese_entretien": documents})
    y = pd.Series([0, 1, 2] * 10, name="classe_retour_emploi")

    table = compare_baseline_hyperparameters(
        model_type="random_forest",
        scenario_name="texte_seul",
        X_train=X,
        y_train=y,
        parameter_sets={
            "petit": {"n_estimators": 10, "max_depth": None},
            "profond": {"n_estimators": 10, "max_depth": 5},
        },
        cv_folds=3,
    )

    assert len(table) == 2
    assert table["selected"].sum() == 1
    assert set(table.columns) >= {
        "f1_macro_mean",
        "f1_macro_std",
        "recall_classe_2_mean",
        "analysis",
    }













