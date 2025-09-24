"""Training entrypoint for the polyreactivity model."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression

from .config import Config, load_config
from .data_loaders import boughter, harvey, jain, shehata
from .data_loaders.utils import deduplicate_sequences
from .features.pipeline import FeaturePipeline, FeaturePipelineState, build_feature_pipeline
from .models.calibrate import fit_calibrator
from .models.linear import LinearModelConfig, TrainedModel, build_estimator, train_linear_model
from .utils.io import write_table
from .utils.logging import configure_logging
from .utils.metrics import bootstrap_metric_intervals, compute_metrics
from .utils.plots import plot_precision_recall, plot_reliability_curve, plot_roc_curve
from .utils.seeds import set_global_seeds

DATASET_LOADERS = {
    "boughter": boughter.load_dataframe,
    "jain": jain.load_dataframe,
    "shehata": shehata.load_dataframe,
    "shehata_curated": shehata.load_dataframe,
    "harvey": harvey.load_dataframe,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train polyreactivity model")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--train", required=True, help="Training dataset path")
    parser.add_argument(
        "--eval",
        nargs="*",
        default=[],
        help="Evaluation dataset paths",
    )
    parser.add_argument(
        "--save-to",
        default="artifacts/model.joblib",
        help="Path to save trained model artifact",
    )
    parser.add_argument(
        "--report-to",
        default="artifacts",
        help="Directory for metrics, predictions, and plots",
    )
    parser.add_argument(
        "--train-loader",
        choices=list(DATASET_LOADERS.keys()),
        help="Optional explicit loader for training dataset",
    )
    parser.add_argument(
        "--eval-loaders",
        nargs="*",
        help="Optional explicit loaders for evaluation datasets (aligned with --eval order)",
    )
    parser.add_argument(
        "--backend",
        choices=["plm", "descriptors", "concat"],
        help="Override feature backend",
    )
    parser.add_argument("--plm-model", help="Override PLM model name")
    parser.add_argument("--cache-dir", help="Override embedding cache directory")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Device override")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for embeddings")
    parser.add_argument(
        "--heavy-only",
        action="store_true",
        default=True,
        help="Use heavy chains only (default true)",
    )
    parser.add_argument(
        "--paired",
        dest="heavy_only",
        action="store_false",
        help="Use paired heavy/light chains when available.",
    )
    parser.add_argument(
        "--include-families",
        nargs="*",
        help="Optional list of family names to retain in the training dataset",
    )
    parser.add_argument(
        "--exclude-families",
        nargs="*",
        help="Optional list of family names to drop from the training dataset",
    )
    parser.add_argument(
        "--include-species",
        nargs="*",
        help="Optional list of species (e.g. human, mouse) to retain",
    )
    parser.add_argument(
        "--cv-group-column",
        default="lineage",
        help="Column name used to group samples during cross-validation (default: lineage)",
    )
    parser.add_argument(
        "--no-group-cv",
        action="store_true",
        help="Disable group-aware cross-validation even if group column is present",
    )
    parser.add_argument(
        "--keep-train-duplicates",
        action="store_true",
        help="Keep duplicate keys within the training dataset when deduplicating across splits",
    )
    parser.add_argument(
        "--dedupe-key-columns",
        nargs="*",
        help="Columns used to detect duplicates across datasets (defaults to heavy/light sequences)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="Number of bootstrap resamples for confidence intervals (0 to disable).",
    )
    parser.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=0.05,
        help="Alpha for two-sided bootstrap confidence intervals (default 0.05 → 95% CI).",
    )
    parser.add_argument(
        "--write-train-in-sample",
        action="store_true",
        help=(
            "Persist in-sample metrics on the full training set; disabled by default to avoid"
            " over-optimistic reporting."
        ),
    )
    return parser


def _infer_loader(path: str, explicit: str | None) -> tuple[str, callable]:
    if explicit:
        return explicit, DATASET_LOADERS[explicit]
    lower = Path(path).stem.lower()
    for name, loader in DATASET_LOADERS.items():
        if name in lower:
            return name, loader
    msg = f"Could not infer loader for dataset: {path}. Provide --train-loader/--eval-loaders."
    raise ValueError(msg)


def _load_dataset(path: str, loader_name: str, loader_fn, *, heavy_only: bool) -> pd.DataFrame:
    frame = loader_fn(path, heavy_only=heavy_only)
    frame["source"] = loader_name
    return frame


def _apply_dataset_filters(
    frame: pd.DataFrame,
    *,
    include_families: Sequence[str] | None,
    exclude_families: Sequence[str] | None,
    include_species: Sequence[str] | None,
) -> pd.DataFrame:
    filtered = frame.copy()
    if include_families:
        families = {fam.lower() for fam in include_families}
        if "family" in filtered.columns:
            filtered = filtered[
                filtered["family"].astype(str).str.lower().isin(families)
            ]
    if exclude_families:
        families_ex = {fam.lower() for fam in exclude_families}
        if "family" in filtered.columns:
            filtered = filtered[
                ~filtered["family"].astype(str).str.lower().isin(families_ex)
            ]
    if include_species:
        species_set = {spec.lower() for spec in include_species}
        if "species" in filtered.columns:
            filtered = filtered[
                filtered["species"].astype(str).str.lower().isin(species_set)
            ]
    return filtered.reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.device:
        config.device = args.device
    if args.backend:
        config.feature_backend.type = args.backend
    if args.cache_dir:
        config.feature_backend.cache_dir = args.cache_dir
    if args.plm_model:
        config.feature_backend.plm_model_name = args.plm_model

    logger = configure_logging()
    set_global_seeds(config.seed)
    _log_environment(logger)

    heavy_only = args.heavy_only

    train_name, train_loader = _infer_loader(args.train, args.train_loader)
    train_df = _load_dataset(args.train, train_name, train_loader, heavy_only=heavy_only)
    train_df = _apply_dataset_filters(
        train_df,
        include_families=args.include_families,
        exclude_families=args.exclude_families,
        include_species=args.include_species,
    )

    eval_frames: list[pd.DataFrame] = []
    if args.eval:
        loaders_iter = args.eval_loaders or []
        for idx, eval_path in enumerate(args.eval):
            explicit = loaders_iter[idx] if idx < len(loaders_iter) else None
            eval_name, eval_loader = _infer_loader(eval_path, explicit)
            eval_df = _load_dataset(eval_path, eval_name, eval_loader, heavy_only=heavy_only)
            eval_frames.append(eval_df)

    all_frames = [train_df, *eval_frames]
    dedup_keep = {0} if args.keep_train_duplicates else set()
    deduped_frames = deduplicate_sequences(
        all_frames,
        heavy_only=heavy_only,
        key_columns=args.dedupe_key_columns,
        keep_intra_frames=dedup_keep,
    )
    train_df = deduped_frames[0]
    eval_frames = deduped_frames[1:]

    pipeline_factory = lambda: build_feature_pipeline(  # noqa: E731
        config,
        backend_override=args.backend,
        plm_model_override=args.plm_model,
        cache_dir_override=args.cache_dir,
    )

    model_config = LinearModelConfig(
        head=config.model.head,
        C=config.model.C,
        class_weight=config.model.class_weight,
    )

    groups = None
    if not args.no_group_cv and args.cv_group_column:
        if args.cv_group_column in train_df.columns:
            groups = train_df[args.cv_group_column].fillna("").astype(str).to_numpy()
        else:
            logger.warning(
                "Group column '%s' not found in training dataframe; falling back to standard CV",
                args.cv_group_column,
            )

    cv_results = _cross_validate(
        train_df,
        pipeline_factory,
        model_config,
        config,
        heavy_only=heavy_only,
        batch_size=args.batch_size,
        groups=groups,
    )

    trained_model, feature_pipeline = _fit_full_model(
        train_df,
        pipeline_factory,
        model_config,
        config,
        heavy_only=heavy_only,
        batch_size=args.batch_size,
    )

    outputs_dir = Path(args.report_to)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    metrics_df, preds_rows = _evaluate_datasets(
        train_df,
        eval_frames,
        trained_model,
        feature_pipeline,
        config,
        cv_results,
        outputs_dir,
        batch_size=args.batch_size,
        heavy_only=heavy_only,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_alpha=args.bootstrap_alpha,
        write_train_in_sample=args.write_train_in_sample,
    )

    write_table(metrics_df, outputs_dir / config.io.metrics_filename)
    preds_df = pd.DataFrame(preds_rows)
    write_table(preds_df, outputs_dir / config.io.preds_filename)

    artifact = {
        "config": config,
        "feature_state": feature_pipeline.get_state(),
        "model": trained_model,
    }
    Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.save_to)

    logger.info("Training complete. Metrics written to %s", outputs_dir)
    return 0


def _cross_validate(
    train_df: pd.DataFrame,
    pipeline_factory,
    model_config: LinearModelConfig,
    config: Config,
    *,
    heavy_only: bool,
    batch_size: int,
    groups: np.ndarray | None = None,
):
    y = train_df["label"].to_numpy(dtype=int)
    n_samples = len(y)
    # Determine a safe number of folds for tiny fixtures; prefer the configured value
    # but never exceed the number of samples. Fall back to non-stratified KFold when
    # per-class counts are too small for stratification (e.g., 1 positive/1 negative).
    n_splits = max(2, min(config.training.cv_folds, n_samples))

    use_stratified = True
    class_counts = np.bincount(y) if y.size else np.array([])
    if class_counts.size > 0 and (class_counts.min(initial=0) < n_splits):
        use_stratified = False

    if groups is not None and use_stratified:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.seed,
        )
        split_iter = splitter.split(train_df, y, groups)
    elif use_stratified:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.seed,
        )
        split_iter = splitter.split(train_df, y)
    else:
        # Non-stratified fallback for extreme class imbalance / tiny datasets
        from sklearn.model_selection import KFold  # local import to limit surface

        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
        split_iter = splitter.split(train_df)
    oof_scores = np.zeros(len(train_df), dtype=float)
    metrics_per_fold: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
        train_slice = train_df.iloc[train_idx].reset_index(drop=True)
        val_slice = train_df.iloc[val_idx].reset_index(drop=True)

        pipeline: FeaturePipeline = pipeline_factory()
        X_train = pipeline.fit_transform(train_slice, heavy_only=heavy_only, batch_size=batch_size)
        X_val = pipeline.transform(val_slice, heavy_only=heavy_only, batch_size=batch_size)

        y_train = y[train_idx]
        y_val = y[val_idx]

        # Handle degenerate folds where training data contains a single class
        if np.unique(y_train).size < 2:
            fallback_prob = float(y.mean()) if y.size else 0.5
            y_scores = np.full(X_val.shape[0], fallback_prob, dtype=float)
        else:
            trained = train_linear_model(
                X_train, y_train, config=model_config, random_state=config.seed
            )
            calibrator = _fit_model_calibrator(
                model_config,
                config,
                X_train,
                y_train,
                base_estimator=trained.estimator,
            )
            trained.calibrator = calibrator
            if calibrator is not None:
                y_scores = calibrator.predict_proba(X_val)[:, 1]
            else:
                y_scores = trained.predict_proba(X_val)
        oof_scores[val_idx] = y_scores

        fold_metrics = compute_metrics(y_val, y_scores)
        try:
            fold_metrics["roc_auc"] = float(roc_auc_score(y_val, y_scores))
        except ValueError:
            # For tiny validation folds with a single class, ROC-AUC is undefined
            pass
        metrics_per_fold.append(fold_metrics)

    metrics_mean: dict[str, float] = {}
    metrics_std: dict[str, float] = {}
    metric_names = list(metrics_per_fold[0].keys()) if metrics_per_fold else []
    for metric in metric_names:
        values = [fold[metric] for fold in metrics_per_fold]
        metrics_mean[metric] = float(np.mean(values))
        metrics_std[metric] = float(np.std(values, ddof=1))

    return {
        "oof_scores": oof_scores,
        "metrics_per_fold": metrics_per_fold,
        "metrics_mean": metrics_mean,
        "metrics_std": metrics_std,
    }


def _fit_full_model(
    train_df: pd.DataFrame,
    pipeline_factory,
    model_config: LinearModelConfig,
    config: Config,
    *,
    heavy_only: bool,
    batch_size: int,
) -> tuple[TrainedModel, FeaturePipeline]:
    pipeline: FeaturePipeline = pipeline_factory()
    X_train = pipeline.fit_transform(train_df, heavy_only=heavy_only, batch_size=batch_size)
    y_train = train_df["label"].to_numpy(dtype=int)

    trained = train_linear_model(X_train, y_train, config=model_config, random_state=config.seed)
    calibrator = _fit_model_calibrator(
        model_config,
        config,
        X_train,
        y_train,
        base_estimator=trained.estimator,
    )
    trained.calibrator = calibrator

    return trained, pipeline


def _evaluate_datasets(
    train_df: pd.DataFrame,
    eval_frames: list[pd.DataFrame],
    trained_model: TrainedModel,
    pipeline: FeaturePipeline,
    config: Config,
    cv_results: dict,
    outputs_dir: Path,
    *,
    batch_size: int,
    heavy_only: bool,
    bootstrap_samples: int,
    bootstrap_alpha: float,
    write_train_in_sample: bool,
):
    metrics_lookup: dict[str, dict[str, float]] = {}
    preds_rows: list[dict[str, float]] = []

    metrics_mean: dict[str, float] = cv_results["metrics_mean"]
    metrics_std: dict[str, float] = cv_results["metrics_std"]

    for metric_name, value in metrics_mean.items():
        metrics_lookup.setdefault(metric_name, {"metric": metric_name})[
            "train_cv_mean"
        ] = value
    for metric_name, value in metrics_std.items():
        metrics_lookup.setdefault(metric_name, {"metric": metric_name})[
            "train_cv_std"
        ] = value

    train_scores = cv_results["oof_scores"]
    train_preds = train_df[["id", "source", "label"]].copy()
    train_preds["y_true"] = train_preds["label"]
    train_preds["y_score"] = train_scores
    train_preds["y_pred"] = (train_scores >= 0.5).astype(int)
    train_preds["split"] = "train_cv_oof"
    preds_rows.extend(
        train_preds[["id", "source", "split", "y_true", "y_score", "y_pred"]].to_dict("records")
    )

    plot_reliability_curve(
        train_preds["y_true"], train_preds["y_score"], path=outputs_dir / "reliability_train.png"
    )
    plot_precision_recall(
        train_preds["y_true"], train_preds["y_score"], path=outputs_dir / "pr_train.png"
    )
    plot_roc_curve(train_preds["y_true"], train_preds["y_score"], path=outputs_dir / "roc_train.png")

    if bootstrap_samples > 0:
        ci_map = bootstrap_metric_intervals(
            train_preds["y_true"],
            train_preds["y_score"],
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            random_state=config.seed,
        )
        for metric_name, stats in ci_map.items():
            row = metrics_lookup.setdefault(metric_name, {"metric": metric_name})
            row["train_cv_ci_lower"] = stats.get("ci_lower")
            row["train_cv_ci_upper"] = stats.get("ci_upper")
            row["train_cv_ci_median"] = stats.get("ci_median")

    if write_train_in_sample:
        train_features_full = pipeline.transform(
            train_df, heavy_only=heavy_only, batch_size=batch_size
        )
        train_full_scores = trained_model.predict_proba(train_features_full)
        train_full_metrics = compute_metrics(
            train_df["label"].to_numpy(dtype=int), train_full_scores
        )
        (outputs_dir / "train_in_sample.json").write_text(
            json.dumps(train_full_metrics, indent=2),
            encoding="utf-8",
        )

    for frame in eval_frames:
        if frame.empty:
            continue
        features = pipeline.transform(frame, heavy_only=heavy_only, batch_size=batch_size)
        scores = trained_model.predict_proba(features)
        y_true = frame["label"].to_numpy(dtype=int)
        metrics = compute_metrics(y_true, scores)
        dataset_name = frame["source"].iloc[0]
        for metric_name, value in metrics.items():
            metrics_lookup.setdefault(metric_name, {"metric": metric_name})[
                dataset_name
            ] = value

        preds = frame[["id", "source", "label"]].copy()
        preds["y_true"] = preds["label"]
        preds["y_score"] = scores
        preds["y_pred"] = (scores >= 0.5).astype(int)
        preds["split"] = dataset_name
        preds_rows.extend(
            preds[["id", "source", "split", "y_true", "y_score", "y_pred"]].to_dict("records")
        )

        plot_reliability_curve(
            preds["y_true"],
            preds["y_score"],
            path=outputs_dir / f"reliability_{dataset_name}.png",
        )
        plot_precision_recall(
            preds["y_true"],
            preds["y_score"],
            path=outputs_dir / f"pr_{dataset_name}.png",
        )
        plot_roc_curve(
            preds["y_true"], preds["y_score"], path=outputs_dir / f"roc_{dataset_name}.png"
        )

        if bootstrap_samples > 0:
            ci_map = bootstrap_metric_intervals(
                preds["y_true"],
                preds["y_score"],
                n_bootstrap=bootstrap_samples,
                alpha=bootstrap_alpha,
                random_state=config.seed,
            )
            for metric_name, stats in ci_map.items():
                row = metrics_lookup.setdefault(metric_name, {"metric": metric_name})
                row[f"{dataset_name}_ci_lower"] = stats.get("ci_lower")
                row[f"{dataset_name}_ci_upper"] = stats.get("ci_upper")
                row[f"{dataset_name}_ci_median"] = stats.get("ci_median")

    metrics_df = pd.DataFrame(sorted(metrics_lookup.values(), key=lambda row: row["metric"]))
    return metrics_df, preds_rows


def _fit_model_calibrator(
    model_config: LinearModelConfig,
    config: Config,
    X: np.ndarray,
    y: np.ndarray,
    *,
    base_estimator: Any | None = None,
):
    method = config.calibration.method
    if not method:
        return None
    if len(np.unique(y)) < 2:
        return None

    if len(y) >= 4:
        cv_cal = min(config.training.cv_folds, max(2, len(y) // 2))
        estimator = build_estimator(config=model_config, random_state=config.seed)
        if isinstance(estimator, LogisticRegression) and X.shape[0] >= 1000:
            estimator.set_params(solver="lbfgs")
        calibrator = fit_calibrator(estimator, X, y, method=method, cv=cv_cal)
    else:
        estimator = base_estimator or build_estimator(config=model_config, random_state=config.seed)
        if isinstance(estimator, LogisticRegression) and X.shape[0] >= 1000:
            estimator.set_params(solver="lbfgs")
        estimator.fit(X, y)
        calibrator = fit_calibrator(estimator, X, y, method=method, cv="prefit")
    return calibrator


def _log_environment(logger) -> None:
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - best effort
        git_head = "unknown"
    try:
        pip_freeze = subprocess.check_output(["pip", "freeze"], text=True)
    except Exception:  # pragma: no cover
        pip_freeze = ""
    logger.info("git_head=%s", git_head)
    logger.info("pip_freeze=%s", pip_freeze)


if __name__ == "__main__":
    raise SystemExit(main())
