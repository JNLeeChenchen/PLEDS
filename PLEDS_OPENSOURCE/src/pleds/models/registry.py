"""Model frontend registry used by application-level PLEDS searches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

from .additive_score import train_and_lower_bernoulli_nb
from .decision_tree import train_and_lower_decision_tree
from .isolation_forest import train_and_lower_isolation_forest
from .piecewise_range import train_and_lower_piecewise_range
from .rule_list import train_and_lower_rule_list
from .staged_tree import lower_tree_to_stages, select_tree_representation
from .tm_guided import train_and_lower_tm_guided
from .tree_ensemble import train_and_lower_random_forest
from .xgboost_ensemble import train_and_lower_xgboost_ensemble
from pleds.p4gen.staged_tree import staged_tree_table_plan


@dataclass(frozen=True)
class ModelFrontendSpec:
    name: str
    output_contracts: tuple[str, ...]
    stochastic: bool
    supported_feature_counts: tuple[int, ...]


ALL_DEPLOYABLE_FEATURE_COUNTS = (12, 104)


APPLICATION_FEATURE_COUNTS = {
    "flow_record_collection": {
        "decision_tree": ALL_DEPLOYABLE_FEATURE_COUNTS,
        "random_forest_ensemble": (12,),
        "xgboost_ensemble": (12,),
        "rule_list": ALL_DEPLOYABLE_FEATURE_COUNTS,
        "naive_bayes_lookup": (12,),
        "piecewise_range": (12,),
        "isolation_forest": (12,),
        "tm_guided": ALL_DEPLOYABLE_FEATURE_COUNTS,
    },
}


MODEL_FRONTENDS = {
    "decision_tree": ModelFrontendSpec(
        "decision_tree", ("binary_decision",), False, ALL_DEPLOYABLE_FEATURE_COUNTS
    ),
    "random_forest_ensemble": ModelFrontendSpec(
        "random_forest_ensemble", ("binary_decision",), True, (12,)
    ),
    "xgboost_ensemble": ModelFrontendSpec(
        "xgboost_ensemble", ("binary_decision",), True, (12,)
    ),
    "rule_list": ModelFrontendSpec(
        "rule_list", ("binary_decision",), False, ALL_DEPLOYABLE_FEATURE_COUNTS
    ),
    "naive_bayes_lookup": ModelFrontendSpec(
        "naive_bayes_lookup", ("binary_decision", "score"), False, (12,)
    ),
    "piecewise_range": ModelFrontendSpec(
        "piecewise_range",
        ("binary_decision", "score", "partition_class"),
        False,
        (12,),
    ),
    "isolation_forest": ModelFrontendSpec(
        "isolation_forest", ("binary_decision", "score"), True, (12,)
    ),
    "tm_guided": ModelFrontendSpec(
        "tm_guided", ("binary_decision", "score"), True, ALL_DEPLOYABLE_FEATURE_COUNTS
    ),
}


@dataclass(frozen=True)
class TrainedModelCandidate:
    candidate_id: str
    family: str
    parameters: dict[str, object]
    lowered: object
    model_plan: dict[str, object]
    report_object: object
    report: dict[str, object]
    output_contracts: tuple[str, ...]
    plan_sha256: str


def available_model_frontends() -> tuple[str, ...]:
    return tuple(sorted(MODEL_FRONTENDS))


def supports_application_feature_count(
    family: str,
    application: str,
    feature_count: int,
) -> bool:
    return feature_count in APPLICATION_FEATURE_COUNTS.get(application, {}).get(
        family, ()
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_id(family: str, parameters: Mapping[str, object], seed: int) -> str:
    digest = _canonical_sha256(
        {"family": family, "parameters": dict(parameters), "seed": seed}
    )[:12]
    return f"{family}-{digest}"


def train_and_lower_candidate(
    family: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    parameters: Mapping[str, object] | None = None,
) -> TrainedModelCandidate:
    """Train one frontend and return its exact deployable/lowered candidate."""

    if family not in MODEL_FRONTENDS:
        raise ValueError(
            f"unsupported model frontend {family!r}; available: "
            + ", ".join(available_model_frontends())
        )
    params = dict(parameters or {})
    feature_count = int(x_train.shape[1])
    supported = MODEL_FRONTENDS[family].supported_feature_counts
    if feature_count not in supported:
        raise ValueError(
            f"{family} has no exact deployable lowering for {feature_count} features; "
            f"supported widths: {supported}"
        )

    if family == "decision_tree":
        teacher, lowered, report = train_and_lower_decision_tree(
            x_train,
            y_train,
            max_depth=int(params.get("max_depth", 5)),
            max_leaf_nodes=(
                None
                if params.get("max_leaf_nodes") is None
                else int(params["max_leaf_nodes"])
            ),
            min_samples_leaf=int(params.get("min_samples_leaf", 10)),
            random_state=seed,
        )
        representation = str(params.get("representation", "root_to_leaf_rules"))
        if representation not in {"root_to_leaf_rules", "staged_nodes", "auto"}:
            raise ValueError(
                f"unsupported decision-tree representation: {representation}"
            )
        if representation in {"staged_nodes", "auto"}:
            staged = lower_tree_to_stages(teacher, feature_count=feature_count)
            if representation == "auto":
                representation = select_tree_representation(
                    lowered,
                    staged,
                    max_stages=int(params.get("representation_max_stages", 8)),
                    max_entries=int(params.get("representation_max_entries", 1024)),
                    stage_weight=int(params.get("representation_stage_weight", 512)),
                )
            if representation == "staged_nodes":
                lowered = staged
                plan = staged.to_model_plan()
                plan["match_action_plan"] = staged_tree_table_plan(plan)
            else:
                plan = lowered.to_rule_plan(feature_count=feature_count)
                plan["selected_representation"] = "root_to_leaf_rules"
        else:
            plan = lowered.to_rule_plan(feature_count=feature_count)
            plan["selected_representation"] = "root_to_leaf_rules"
    elif family == "random_forest_ensemble":
        _teacher, lowered, report = train_and_lower_random_forest(
            x_train,
            y_train,
            n_estimators=int(params.get("n_estimators", 5)),
            max_depth=int(params.get("max_depth", 3)),
            max_leaf_nodes=(
                None
                if params.get("max_leaf_nodes") is None
                else int(params.get("max_leaf_nodes", 8))
            ),
            min_samples_leaf=int(params.get("min_samples_leaf", 10)),
            random_state=seed,
        )
        plan = lowered.to_model_plan(feature_count=feature_count)
    elif family == "xgboost_ensemble":
        _teacher, lowered, report = train_and_lower_xgboost_ensemble(
            x_train,
            y_train,
            n_estimators=int(params.get("n_estimators", 7)),
            max_depth=int(params.get("max_depth", 2)),
            learning_rate=float(params.get("learning_rate", 0.25)),
            random_state=seed,
        )
        plan = lowered.to_model_plan()
    elif family == "rule_list":
        _teacher, lowered, report = train_and_lower_rule_list(
            x_train,
            y_train,
            max_rules=int(params.get("max_rules", 128)),
        )
        plan = lowered.to_model_plan()
    elif family == "naive_bayes_lookup":
        _teacher, lowered, report = train_and_lower_bernoulli_nb(
            x_train,
            y_train,
            scale=int(params.get("scale", 1024)),
        )
        plan = lowered.to_model_plan()
    elif family == "piecewise_range":
        _teacher, lowered, report = train_and_lower_piecewise_range(
            x_train,
            y_train,
            bucket_count=int(params.get("bucket_count", 4)),
        )
        plan = lowered.to_model_plan()
    elif family == "isolation_forest":
        _teacher, lowered, report = train_and_lower_isolation_forest(
            x_train,
            y_train,
            n_estimators=int(params.get("n_estimators", 75)),
            threshold_quantile=float(params.get("threshold_quantile", 0.8)),
            random_state=seed,
        )
        plan = lowered.to_model_plan()
    elif family == "tm_guided":
        _teacher, lowered, report = train_and_lower_tm_guided(
            x_train,
            y_train,
            random_state=seed,
            number_of_clauses=int(params.get("number_of_clauses", 128)),
            threshold=int(params.get("threshold", 32)),
            specificity=float(params.get("specificity", 5.0)),
            epochs=int(params.get("epochs", 5)),
            max_rules=int(params.get("max_rules", 128)),
            node_sample_size=int(params.get("node_sample_size", 256)),
            max_depth=(
                None if params.get("max_depth") is None else int(params["max_depth"])
            ),
        )
        plan = lowered.to_model_plan()
    else:
        raise ValueError(f"unsupported model frontend: {family}")

    report_payload = report.as_dict()
    plan_hash = _canonical_sha256(plan)
    return TrainedModelCandidate(
        candidate_id=_candidate_id(family, params, seed),
        family=family,
        parameters=params,
        lowered=lowered,
        model_plan=plan,
        report_object=report,
        report=report_payload,
        output_contracts=MODEL_FRONTENDS[family].output_contracts,
        plan_sha256=plan_hash,
    )


__all__ = [
    "MODEL_FRONTENDS",
    "APPLICATION_FEATURE_COUNTS",
    "ModelFrontendSpec",
    "TrainedModelCandidate",
    "available_model_frontends",
    "supports_application_feature_count",
    "train_and_lower_candidate",
]
