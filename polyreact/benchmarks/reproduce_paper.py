"""Reproduce key metrics and visualisations for the polyreactivity model."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from polyreact import train as train_module
from polyreact.config import load_config
from polyreact.features.anarsi import AnarciNumberer
from polyreact.features.pipeline import FeaturePipeline
from polyreact.models.ordinal import (
    fit_negative_binomial_model,
    fit_poisson_model,
    pearson_dispersion,
    regression_metrics,
)


@dataclass(slots=True)
class DatasetSpec:
    name: str
    path: Path
    display: str


DISPLAY_LABELS = {
    "jain": "Jain (2017)",
    "shehata": "Shehata PSR (398)",
    "shehata_curated": "Shehata curated (88)",
    "harvey": "Harvey (2022)",
}

RAW_LABELS = {
    "jain": "jain2017",
    "shehata": "shehata2019",
    "shehata_curated": "shehata2019_curated",
    "harvey": "harvey2022",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce paper-style metrics and plots")
    parser.add_argument(
        "--train-data",
        default="data/processed/boughter_counts_rebuilt.csv",
        help="Reconstructed Boughter dataset path.",
    )
    parser.add_argument(
        "--full-data",
        default="data/processed/boughter_counts_rebuilt.csv",
        help="Dataset (including mild flags) for correlation analysis.",
    )
    parser.add_argument("--jain", default="data/processed/jain.csv")
    parser.add_argument(
        "--shehata",
        default="data/processed/shehata_full.csv",
        help="Full Shehata PSR panel (398 sequences) in processed CSV form.",
    )
    parser.add_argument(
        "--shehata-curated",
        default="data/processed/shehata_curated.csv",
        help="Optional curated subset of Shehata et al. (88 sequences).",
    )
    parser.add_argument("--harvey", default="data/processed/harvey.csv")
    parser.add_argument("--output-dir", default="artifacts/paper")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap resamples for metrics confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=0.05,
        help="Alpha for bootstrap confidence intervals (default 0.05 → 95%).",
    )
    parser.add_argument(
        "--human-only",
        action="store_true",
        help=(
            "Restrict the main cross-validation run to human HIV and influenza families"
            " (legacy behaviour). By default all Boughter families, including mouse IgA,"
            " participate in CV as in Sakhnini et al."
        ),
    )
    parser.add_argument(
        "--skip-flag-regression",
        action="store_true",
        help="Skip ELISA flag regression diagnostics (Poisson/NB).",
    )
    parser.add_argument(
        "--skip-lofo",
        action="store_true",
        help="Skip leave-one-family-out experiments.",
    )
    parser.add_argument(
        "--skip-descriptor-variants",
        action="store_true",
        help="Skip descriptor-only benchmark variants.",
    )
    parser.add_argument(
        "--skip-fragment-variants",
        action="store_true",
        help="Skip CDR fragment ablation benchmarks.",
    )
    return parser


def _config_to_dict(config) -> Dict[str, Any]:
    data = asdict(config)
    data.pop("raw", None)
    return data


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result.get(key, {}), value)
        else:
            result[key] = value
    return result


def _write_variant_config(
    base_config: Dict[str, Any],
    overrides: Dict[str, Any],
    target_path: Path,
) -> Path:
    merged = _deep_merge(base_config, overrides)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, sort_keys=False)
    return target_path


def _collect_metric_records(variant: str, metrics: pd.DataFrame) -> list[dict[str, Any]]:
    tracked = {
        "roc_auc",
        "pr_auc",
        "accuracy",
        "f1",
        "f1_positive",
        "f1_negative",
        "precision",
        "sensitivity",
        "specificity",
        "brier",
        "ece",
        "mce",
    }
    records: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        metric_name = row["metric"]
        if metric_name not in tracked:
            continue
        record = {"variant": variant, "metric": metric_name}
        for column in metrics.columns:
            if column == "metric":
                continue
            record[column] = float(row[column]) if pd.notna(row[column]) else np.nan
        records.append(record)
    return records


def _dump_coefficients(model_path: Path, output_path: Path) -> None:
    artifact = joblib.load(model_path)
    trained = artifact["model"]
    estimator = getattr(trained, "estimator", None)
    if estimator is None or not hasattr(estimator, "coef_"):
        return
    coefs = estimator.coef_[0]
    feature_state = artifact.get("feature_state")
    feature_names: list[str]
    if feature_state is not None and getattr(feature_state, "feature_names", None):
        feature_names = list(feature_state.feature_names)
    else:
        feature_names = [f"f{i}" for i in range(len(coefs))]
    coeff_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coefs,
            "abs_coef": np.abs(coefs),
        }
    ).sort_values("abs_coef", ascending=False)
    coeff_df.to_csv(output_path, index=False)


def _summarise_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split, group in preds.groupby("split"):
        stats = {
            "split": split,
            "n_samples": int(len(group)),
            "positives": int(group["y_true"].sum()),
            "positive_rate": float(group["y_true"].mean()) if len(group) else np.nan,
            "score_mean": float(group["y_score"].mean()) if len(group) else np.nan,
            "score_std": float(group["y_score"].std(ddof=1)) if len(group) > 1 else np.nan,
        }
        records.append(stats)
    return pd.DataFrame(records)


def _summarise_raw_dataset(path: Path, name: str) -> dict[str, Any]:
    df = pd.read_csv(path)
    summary: dict[str, Any] = {
        "dataset": name,
        "path": str(path),
        "rows": int(len(df)),
    }
    if "label" in df.columns:
        positives = int(df["label"].sum())
        summary["positives"] = positives
        summary["positive_rate"] = float(df["label"].mean()) if len(df) else np.nan
    if "reactivity_count" in df.columns:
        summary["reactivity_count_mean"] = float(df["reactivity_count"].mean())
        summary["reactivity_count_median"] = float(df["reactivity_count"].median())
        summary["reactivity_count_max"] = int(df["reactivity_count"].max())
    if "smp" in df.columns:
        summary["smp_mean"] = float(df["smp"].mean())
        summary["smp_median"] = float(df["smp"].median())
        summary["smp_max"] = float(df["smp"].max())
        summary["smp_min"] = float(df["smp"].min())
    summary["unique_heavy"] = int(df["heavy_seq"].nunique()) if "heavy_seq" in df.columns else np.nan
    return summary


def _extract_region_sequence(sequence: str, regions: List[str], numberer: AnarciNumberer) -> str:
    if not sequence:
        return ""
    upper_regions = [region.upper() for region in regions]
    if upper_regions == ["VH"]:
        return sequence
    try:
        numbered = numberer.number_sequence(sequence)
    except Exception:
        return ""
    fragments: list[str] = []
    for region in upper_regions:
        if region == "VH":
            return sequence
        fragment = numbered.regions.get(region)
        if not fragment:
            return ""
        fragments.append(fragment)
    return "".join(fragments)


def _make_region_dataset(
    frame: pd.DataFrame, regions: List[str], numberer: AnarciNumberer
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    dropped = 0
    for record in frame.to_dict(orient="records"):
        new_seq = _extract_region_sequence(record.get("heavy_seq", ""), regions, numberer)
        if not new_seq:
            dropped += 1
            continue
        updated = record.copy()
        updated["heavy_seq"] = new_seq
        updated["light_seq"] = ""
        records.append(updated)
    result = pd.DataFrame(records, columns=frame.columns)
    summary = {
        "regions": "+".join(regions),
        "input_rows": int(len(frame)),
        "retained_rows": int(len(result)),
        "dropped_rows": int(dropped),
    }
    return result, summary


def run_train(
    *,
    train_path: Path,
    eval_specs: Sequence[DatasetSpec],
    output_dir: Path,
    model_path: Path,
    config: str,
    batch_size: int,
    include_species: list[str] | None = None,
    include_families: list[str] | None = None,
    exclude_families: list[str] | None = None,
    keep_duplicates: bool = False,
    group_column: str | None = "lineage",
    train_loader: str | None = None,
    bootstrap_samples: int = 200,
    bootstrap_alpha: float = 0.05,
) -> None:
    args: list[str] = [
        "--config",
        str(config),
        "--train",
        str(train_path),
        "--report-to",
        str(output_dir),
        "--save-to",
        str(model_path),
        "--batch-size",
        str(batch_size),
    ]

    if eval_specs:
        args.append("--eval")
        args.extend(str(spec.path) for spec in eval_specs)

    if train_loader:
        args.extend(["--train-loader", train_loader])
    if eval_specs:
        args.append("--eval-loaders")
        args.extend(spec.name for spec in eval_specs)
    if include_species:
        args.append("--include-species")
        args.extend(include_species)
    if include_families:
        args.append("--include-families")
        args.extend(include_families)
    if exclude_families:
        args.append("--exclude-families")
        args.extend(exclude_families)
    if keep_duplicates:
        args.append("--keep-train-duplicates")
    if group_column:
        args.extend(["--cv-group-column", group_column])
    else:
        args.append("--no-group-cv")
    args.extend(["--bootstrap-samples", str(bootstrap_samples)])
    args.extend(["--bootstrap-alpha", str(bootstrap_alpha)])

    exit_code = train_module.main(args)
    if exit_code != 0:
        raise RuntimeError(f"Training command failed with exit code {exit_code}")


def compute_spearman(model_path: Path, dataset_path: Path, batch_size: int) -> tuple[float, float, pd.DataFrame]:
    artifact = joblib.load(model_path)
    config = artifact["config"]
    pipeline_state = artifact["feature_state"]
    trained_model = artifact["model"]

    pipeline = FeaturePipeline(backend=config.feature_backend, descriptors=config.descriptors, device=config.device)
    pipeline.load_state(pipeline_state)

    dataset = pd.read_csv(dataset_path)
    features = pipeline.transform(dataset, heavy_only=True, batch_size=batch_size)
    scores = trained_model.predict_proba(features)
    dataset = dataset.copy()
    dataset["score"] = scores

    stat, pvalue = spearmanr(dataset["reactivity_count"], dataset["score"])
    return float(stat), float(pvalue), dataset


def plot_accuracy(
    metrics: pd.DataFrame,
    output_path: Path,
    eval_specs: Sequence[DatasetSpec],
) -> None:
    row = metrics.loc[metrics["metric"] == "accuracy"].iloc[0]
    labels = ["Train CV"] + [spec.display for spec in eval_specs]
    values = [row.get("train_cv_mean", np.nan)] + [row.get(spec.name, np.nan) for spec in eval_specs]

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(labels))
    ax.bar(xs, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_xticks(xs, labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Polyreactivity accuracy overview")
    for x, val in zip(xs, values, strict=False):
        if np.isnan(val):
            continue
        ax.text(x, val + 0.02, f"{val:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_rocs(
    preds: pd.DataFrame,
    output_path: Path,
    eval_specs: Sequence[DatasetSpec],
) -> None:
    mapping = {"train_cv_oof": "Train CV"}
    for spec in eval_specs:
        mapping[spec.name] = spec.display
    fig, ax = plt.subplots(figsize=(6, 6))
    for split, label in mapping.items():
        subset = preds[preds["split"] == split]
        if subset.empty:
            continue
        fpr, tpr, _ = roc_curve(subset["y_true"], subset["y_score"])
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_flags_scatter(data: pd.DataFrame, spearman_stat: float, output_path: Path) -> None:
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.1, 0.1, size=len(data))
    x = data["reactivity_count"].to_numpy(dtype=float) + jitter
    y = data["score"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.5, s=10)
    ax.set_xlabel("ELISA flag count")
    ax.set_ylabel("Predicted probability")
    ax.set_title(f"Prediction vs flag count (Spearman={spearman_stat:.2f})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_lofo(
    full_df: pd.DataFrame,
    *,
    families: list[str],
    config: str,
    batch_size: int,
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_alpha: float,
) -> pd.DataFrame:
    results: list[dict[str, float]] = []
    for family in families:
        family_lower = family.lower()
        holdout = full_df[full_df["family"].str.lower() == family_lower].copy()
        train = full_df[full_df["family"].str.lower() != family_lower].copy()
        if holdout.empty or train.empty:
            continue

        train_path = output_dir / f"train_lofo_{family_lower}.csv"
        holdout_path = output_dir / f"eval_lofo_{family_lower}.csv"
        train.to_csv(train_path, index=False)
        holdout.to_csv(holdout_path, index=False)

        run_dir = output_dir / f"lofo_{family_lower}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.joblib"

        run_train(
            train_path=train_path,
            eval_specs=[
                DatasetSpec(
                    name="boughter",
                    path=holdout_path,
                    display=f"{family.title()} holdout",
                )
            ],
            output_dir=run_dir,
            model_path=model_path,
            config=config,
            batch_size=batch_size,
            keep_duplicates=True,
            include_species=None,
            include_families=None,
            exclude_families=None,
            group_column="lineage",
            train_loader="boughter",
            bootstrap_samples=bootstrap_samples,
            bootstrap_alpha=bootstrap_alpha,
        )

        metrics = pd.read_csv(run_dir / "metrics.csv")
        evaluation_cols = [
            col
            for col in metrics.columns
            if col not in {"metric", "train_cv_mean", "train_cv_std"}
        ]
        if not evaluation_cols:
            continue
        eval_col = evaluation_cols[0]
        def _metric_value(name: str) -> float:
            series = metrics.loc[metrics["metric"] == name, eval_col]
            return float(series.values[0]) if not series.empty else float("nan")

        results.append(
            {
                "family": family,
                "accuracy": _metric_value("accuracy"),
                "roc_auc": _metric_value("roc_auc"),
                "pr_auc": _metric_value("pr_auc"),
                "sensitivity": _metric_value("sensitivity"),
                "specificity": _metric_value("specificity"),
            }
        )

    return pd.DataFrame(results)



def run_flag_regression(
    train_path: Path,
    *,
    output_dir: Path,
    config_path: str,
    batch_size: int,
    n_splits: int = 5,
) -> None:
    df = pd.read_csv(train_path)
    if "reactivity_count" not in df.columns:
        return

    config = load_config(config_path)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=config.seed)

    metrics_rows: list[dict[str, float]] = []
    preds_rows: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df), start=1):
        train_split = df.iloc[train_idx].reset_index(drop=True)
        val_split = df.iloc[val_idx].reset_index(drop=True)

        pipeline = FeaturePipeline(
            backend=config.feature_backend,
            descriptors=config.descriptors,
            device=config.device,
        )
        X_train = pipeline.fit_transform(train_split, heavy_only=True, batch_size=batch_size)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        y_train = train_split["reactivity_count"].to_numpy(dtype=float)
        # Train a logistic head to obtain probabilities as a 1-D feature
        clf = LogisticRegression(
            C=config.model.C,
            class_weight=config.model.class_weight,
            max_iter=2000,
            solver="lbfgs",
        )
        clf.fit(X_train_scaled, train_split["label"].to_numpy(dtype=int))
        prob_train = clf.predict_proba(X_train_scaled)[:, 1]

        X_val = pipeline.transform(val_split, heavy_only=True, batch_size=batch_size)
        X_val_scaled = scaler.transform(X_val)
        y_val = val_split["reactivity_count"].to_numpy(dtype=float)
        prob_val = clf.predict_proba(X_val_scaled)[:, 1]

        poisson_X_train = prob_train.reshape(-1, 1)
        poisson_X_val = prob_val.reshape(-1, 1)
        model = fit_poisson_model(poisson_X_train, y_train)
        poisson_preds = model.predict(poisson_X_val)

        n_params = poisson_X_train.shape[1] + 1  # include intercept
        dof = max(len(y_val) - n_params, 1)
        variance_to_mean = float(np.var(y_val, ddof=1) / np.mean(y_val)) if np.mean(y_val) else float("nan")

        spearman_val = float(spearmanr(y_val, poisson_preds).statistic)
        try:
            pearson_val = float(pearsonr(y_val, poisson_preds)[0])
        except Exception:  # pragma: no cover - fallback if correlation fails
            pearson_val = float("nan")

        poisson_metrics = regression_metrics(y_val, poisson_preds)
        poisson_metrics.update(
            {
                "spearman": spearman_val,
                "pearson": pearson_val,
                "pearson_dispersion": pearson_dispersion(y_val, poisson_preds, dof=dof),
                "variance_to_mean": variance_to_mean,
                "fold": fold_idx,
                "model": "poisson",
                "status": "ok",
            }
        )
        metrics_rows.append(poisson_metrics)

        nb_preds: np.ndarray | None = None
        nb_model = None
        try:
            nb_model = fit_negative_binomial_model(poisson_X_train, y_train)
            nb_preds = nb_model.predict(poisson_X_val)
            if not np.all(np.isfinite(nb_preds)):
                raise ValueError("negative binomial produced non-finite predictions")
        except Exception:
            nb_metrics = {
                "spearman": float("nan"),
                "pearson": float("nan"),
                "pearson_dispersion": float("nan"),
                "variance_to_mean": variance_to_mean,
                "alpha": float("nan"),
                "fold": fold_idx,
                "model": "negative_binomial",
                "status": "failed",
            }
            metrics_rows.append(nb_metrics)
        else:
            spearman_nb = float(spearmanr(y_val, nb_preds).statistic)
            try:
                pearson_nb = float(pearsonr(y_val, nb_preds)[0])
            except Exception:  # pragma: no cover
                pearson_nb = float("nan")

            nb_metrics = regression_metrics(y_val, nb_preds)
            nb_metrics.update(
                {
                    "spearman": spearman_nb,
                    "pearson": pearson_nb,
                    "pearson_dispersion": pearson_dispersion(y_val, nb_preds, dof=dof),
                    "variance_to_mean": variance_to_mean,
                    "alpha": nb_model.alpha,
                    "fold": fold_idx,
                    "model": "negative_binomial",
                    "status": "ok",
                }
            )
            metrics_rows.append(nb_metrics)

        records = list(val_split.itertuples(index=False))
        for idx, row in enumerate(records):
            row_id = getattr(row, "id", idx)
            y_true_val = float(getattr(row, "reactivity_count"))
            preds_rows.append(
                {
                    "fold": fold_idx,
                    "model": "poisson",
                    "id": row_id,
                    "y_true": y_true_val,
                    "y_pred": float(poisson_preds[idx]),
                }
            )
            if nb_preds is not None:
                preds_rows.append(
                    {
                        "fold": fold_idx,
                        "model": "negative_binomial",
                        "id": row_id,
                        "y_true": y_true_val,
                        "y_pred": float(nb_preds[idx]),
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "flag_regression_folds.csv", index=False)

    summary_records: list[dict[str, float]] = []
    for model_name, group in metrics_df.groupby("model"):
        for column in group.columns:
            if column in {"fold", "model", "status"}:
                continue
            values = group[column].dropna()
            if values.empty:
                continue
            summary_records.append(
                {
                    "model": model_name,
                    "metric": column,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                }
            )
    if summary_records:
        pd.DataFrame(summary_records).to_csv(
            output_dir / "flag_regression_metrics.csv", index=False
        )

    if preds_rows:
        pd.DataFrame(preds_rows).to_csv(output_dir / "flag_regression_preds.csv", index=False)

def run_descriptor_variants(
    base_config: Dict[str, Any],
    *,
    train_path: Path,
    eval_specs: Sequence[DatasetSpec],
    output_dir: Path,
    batch_size: int,
    include_species: List[str] | None,
    include_families: List[str] | None,
    bootstrap_samples: int,
    bootstrap_alpha: float,
) -> None:
    variants = [
        (
            "descriptors_full_vh",
            {
                "feature_backend": {"type": "descriptors"},
                "descriptors": {
                    "use_anarci": True,
                    "regions": ["CDRH1", "CDRH2", "CDRH3"],
                    "features": [
                        "length",
                        "charge",
                        "hydropathy",
                        "aromaticity",
                        "pI",
                        "net_charge",
                    ],
                },
            },
        ),
        (
            "descriptors_cdrh3_pi",
            {
                "feature_backend": {"type": "descriptors"},
                "descriptors": {
                    "use_anarci": True,
                    "regions": ["CDRH3"],
                    "features": ["pI"],
                },
            },
        ),
        (
            "descriptors_cdrh3_top5",
            {
                "feature_backend": {"type": "descriptors"},
                "descriptors": {
                    "use_anarci": True,
                    "regions": ["CDRH3"],
                    "features": [
                        "pI",
                        "net_charge",
                        "charge",
                        "hydropathy",
                        "length",
                    ],
                },
            },
        ),
    ]

    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    summary_records: list[dict[str, Any]] = []

    for name, overrides in variants:
        variant_config_path = _write_variant_config(
            base_config,
            overrides,
            configs_dir / f"{name}.yaml",
        )
        variant_output = output_dir / name
        variant_output.mkdir(parents=True, exist_ok=True)
        model_path = variant_output / "model.joblib"

        run_train(
            train_path=train_path,
            eval_specs=eval_specs,
            output_dir=variant_output,
            model_path=model_path,
            config=str(variant_config_path),
            batch_size=batch_size,
            include_species=include_species,
            include_families=include_families,
            keep_duplicates=True,
            group_column="lineage",
            train_loader="boughter",
            bootstrap_samples=bootstrap_samples,
            bootstrap_alpha=bootstrap_alpha,
        )

        metrics_path = variant_output / "metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            summary_records.extend(_collect_metric_records(name, metrics_df))

        _dump_coefficients(model_path, variant_output / "coefficients.csv")

    if summary_records:
        pd.DataFrame(summary_records).to_csv(output_dir / "summary.csv", index=False)


def run_fragment_variants(
    config_path: str,
    *,
    train_path: Path,
    eval_specs: Sequence[DatasetSpec],
    output_dir: Path,
    batch_size: int,
    include_species: List[str] | None,
    include_families: List[str] | None,
    bootstrap_samples: int,
    bootstrap_alpha: float,
) -> None:
    numberer = AnarciNumberer()
    specs = [
        ("vh_full", ["VH"]),
        ("cdrh1", ["CDRH1"]),
        ("cdrh2", ["CDRH2"]),
        ("cdrh3", ["CDRH3"]),
        ("cdrh123", ["CDRH1", "CDRH2", "CDRH3"]),
    ]

    summary_rows: list[dict[str, Any]] = []
    metric_summary_rows: list[dict[str, Any]] = []

    for name, regions in specs:
        variant_dir = output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = variant_dir / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        train_df = pd.read_csv(train_path)
        train_variant, train_summary = _make_region_dataset(train_df, regions, numberer)
        train_variant_path = dataset_dir / "train.csv"
        train_variant.to_csv(train_variant_path, index=False)

        eval_variant_specs: list[DatasetSpec] = []
        for spec in eval_specs:
            eval_df = pd.read_csv(spec.path)
            transformed, eval_summary = _make_region_dataset(eval_df, regions, numberer)
            eval_path = dataset_dir / f"{spec.name}.csv"
            transformed.to_csv(eval_path, index=False)
            eval_variant_specs.append(
                DatasetSpec(name=spec.name, path=eval_path, display=spec.display)
            )
            eval_summary.update({"variant": name, "dataset": spec.name})
            summary_rows.append(eval_summary)

        train_summary.update({"variant": name, "dataset": "train"})
        summary_rows.append(train_summary)

        run_train(
            train_path=train_variant_path,
            eval_specs=eval_variant_specs,
            output_dir=variant_dir,
            model_path=variant_dir / "model.joblib",
            config=config_path,
            batch_size=batch_size,
            include_species=include_species,
            include_families=include_families,
            keep_duplicates=True,
            group_column="lineage",
            train_loader="boughter",
            bootstrap_samples=bootstrap_samples,
            bootstrap_alpha=bootstrap_alpha,
        )

        metrics_path = variant_dir / "metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            metric_records = _collect_metric_records(name, metrics_df)
            for record in metric_records:
                record["variant_type"] = "fragment"
            metric_summary_rows.extend(metric_records)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_dir / "fragment_dataset_summary.csv", index=False)
    if metric_summary_rows:
        pd.DataFrame(metric_summary_rows).to_csv(output_dir / "fragment_metrics_summary.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild:
        rebuild_cmd = [
            "python",
            "scripts/rebuild_boughter_from_counts.py",
            "--output",
            str(args.train_data),
        ]
        if subprocess.run(rebuild_cmd, check=False).returncode != 0:
            raise RuntimeError("Dataset rebuild failed")

    train_path = Path(args.train_data)

    def _make_spec(name: str, path_str: str) -> DatasetSpec | None:
        path = Path(path_str)
        if not path.exists():
            return None
        display = DISPLAY_LABELS.get(name, name.replace("_", " ").title())
        return DatasetSpec(name=name, path=path, display=display)

    eval_specs: list[DatasetSpec] = []
    seen_paths: set[Path] = set()
    for name, path_str in [
        ("jain", args.jain),
        ("shehata", args.shehata),
        ("shehata_curated", args.shehata_curated),
        ("harvey", args.harvey),
    ]:
        spec = _make_spec(name, path_str)
        if spec is not None:
            resolved = spec.path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            eval_specs.append(spec)

    base_config = load_config(args.config)
    base_config_dict = _config_to_dict(base_config)

    main_output = output_dir / "main"
    main_output.mkdir(parents=True, exist_ok=True)
    model_path = main_output / "model.joblib"

    main_include_species = ["human"] if args.human_only else None
    main_include_families = ["hiv", "influenza"] if args.human_only else None

    run_train(
        train_path=train_path,
        eval_specs=eval_specs,
        output_dir=main_output,
        model_path=model_path,
        config=args.config,
        batch_size=args.batch_size,
        include_species=main_include_species,
        include_families=main_include_families,
        keep_duplicates=True,
        group_column="lineage",
        train_loader="boughter",
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_alpha=args.bootstrap_alpha,
    )

    metrics = pd.read_csv(main_output / "metrics.csv")
    preds = pd.read_csv(main_output / "preds.csv")

    plot_accuracy(metrics, main_output / "accuracy_overview.png", eval_specs)
    plot_rocs(preds, main_output / "roc_overview.png", eval_specs)

    if not args.skip_flag_regression:
        run_flag_regression(
            train_path=train_path,
            output_dir=main_output,
            config_path=args.config,
            batch_size=args.batch_size,
        )

    split_summary = _summarise_predictions(preds)
    split_summary.to_csv(main_output / "dataset_split_summary.csv", index=False)

    spearman_stat, spearman_p, corr_df = compute_spearman(
        model_path=model_path,
        dataset_path=Path(args.full_data),
        batch_size=args.batch_size,
    )
    plot_flags_scatter(corr_df, spearman_stat, main_output / "prob_vs_flags.png")
    (main_output / "spearman_flags.json").write_text(
        json.dumps({"spearman": spearman_stat, "p_value": spearman_p}, indent=2)
    )
    corr_df.to_csv(main_output / "prob_vs_flags.csv", index=False)

    if not args.skip_lofo:
        full_df = pd.read_csv(args.train_data)
        lofo_dir = output_dir / "lofo_runs"
        lofo_dir.mkdir(parents=True, exist_ok=True)
        lofo_df = run_lofo(
            full_df,
            families=["influenza", "hiv", "mouse_iga"],
            config=args.config,
            batch_size=args.batch_size,
            output_dir=lofo_dir,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_alpha=args.bootstrap_alpha,
        )
        lofo_df.to_csv(output_dir / "lofo_metrics.csv", index=False)

    if not args.skip_descriptor_variants:
        descriptor_dir = output_dir / "descriptor_variants"
        descriptor_dir.mkdir(parents=True, exist_ok=True)
        run_descriptor_variants(
            base_config_dict,
            train_path=train_path,
            eval_specs=eval_specs,
            output_dir=descriptor_dir,
            batch_size=args.batch_size,
            include_species=main_include_species,
            include_families=main_include_families,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_alpha=args.bootstrap_alpha,
        )

    if not args.skip_fragment_variants:
        fragment_dir = output_dir / "fragment_variants"
        fragment_dir.mkdir(parents=True, exist_ok=True)
        run_fragment_variants(
            args.config,
            train_path=train_path,
            eval_specs=eval_specs,
            output_dir=fragment_dir,
            batch_size=args.batch_size,
            include_species=main_include_species,
            include_families=main_include_families,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_alpha=args.bootstrap_alpha,
        )

    raw_summaries = []
    raw_summaries.append(_summarise_raw_dataset(train_path, "boughter_rebuilt"))
    for spec in eval_specs:
        summary_name = RAW_LABELS.get(spec.name, spec.name)
        raw_summaries.append(_summarise_raw_dataset(spec.path, summary_name))
    pd.DataFrame(raw_summaries).to_csv(output_dir / "raw_dataset_summary.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
