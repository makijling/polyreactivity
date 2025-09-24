"""Configuration helpers for the polyreactivity project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

try:
    import importlib.resources as pkg_resources
    from importlib.resources.abc import Traversable
except (ModuleNotFoundError, AttributeError):  # pragma: no cover - compatibility
    import importlib_resources as pkg_resources  # type: ignore[no-redef]
    from importlib_resources.abc import Traversable  # type: ignore[assignment]
from pathlib import Path
from typing import Any, Sequence

import yaml


@dataclass(slots=True)
class FeatureBackendSettings:
    type: str = "plm"
    plm_model_name: str = "facebook/esm2_t12_35M_UR50D"
    layer_pool: str = "mean"
    cache_dir: str = ".cache/embeddings"
    standardize: bool = True


@dataclass(slots=True)
class DescriptorSettings:
    use_anarci: bool = True
    regions: Sequence[str] = field(default_factory=lambda: ["CDRH1", "CDRH2", "CDRH3"])
    features: Sequence[str] = field(
        default_factory=lambda: [
            "length",
            "charge",
            "hydropathy",
            "aromaticity",
            "pI",
            "net_charge",
        ]
    )
    ph: float = 7.4


@dataclass(slots=True)
class ModelSettings:
    head: str = "logreg"
    C: float = 1.0
    class_weight: Any = "balanced"


@dataclass(slots=True)
class CalibrationSettings:
    method: str | None = "isotonic"


@dataclass(slots=True)
class TrainingSettings:
    cv_folds: int = 10
    scoring: str = "roc_auc"
    n_jobs: int = -1


@dataclass(slots=True)
class IOSettings:
    outputs_dir: str = "artifacts"
    preds_filename: str = "preds.csv"
    metrics_filename: str = "metrics.csv"


@dataclass(slots=True)
class Config:
    seed: int = 42
    device: str = "auto"
    feature_backend: FeatureBackendSettings = field(default_factory=FeatureBackendSettings)
    descriptors: DescriptorSettings = field(default_factory=DescriptorSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    io: IOSettings = field(default_factory=IOSettings)

    raw: dict[str, Any] = field(default_factory=dict)


def _merge_section(default: Any, data: dict[str, Any] | None) -> Any:
    if data is None:
        return default
    merged = asdict(default) | data
    return type(default)(**merged)


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML configuration file into a strongly-typed ``Config`` object."""

    data = _read_config_data(path)

    feature_backend = _merge_section(FeatureBackendSettings(), data.get("feature_backend"))
    descriptors = _merge_section(DescriptorSettings(), data.get("descriptors"))
    model = _merge_section(ModelSettings(), data.get("model"))
    calibration = _merge_section(CalibrationSettings(), data.get("calibration"))
    training = _merge_section(TrainingSettings(), data.get("training"))
    io_settings = _merge_section(IOSettings(), data.get("io"))

    config = Config(
        seed=int(data.get("seed", 42)),
        device=str(data.get("device", "auto")),
        feature_backend=feature_backend,
        descriptors=descriptors,
        model=model,
        calibration=calibration,
        training=training,
        io=io_settings,
        raw=data,
    )
    return config


def _read_config_data(path: str | Path | None) -> dict[str, Any]:
    """Return mapping data from YAML or the bundled default."""

    if path is None:
        resource = pkg_resources.files("polyreact.configs") / "default.yaml"
        return _load_yaml_resource(resource)

    resolved = _resolve_config_path(Path(path))
    if resolved is not None:
        return _load_yaml_file(resolved)

    resource_root = pkg_resources.files("polyreact")
    resource = resource_root / Path(path).as_posix()
    if resource.is_file():
        return _load_yaml_resource(resource)

    msg = f"Configuration file not found: {path}"
    raise FileNotFoundError(msg)


def _resolve_config_path(path: Path) -> Path | None:
    if path.exists():
        return path

    if not path.is_absolute():
        candidate = Path(__file__).resolve().parent / path
        if candidate.exists():
            return candidate

    return None


def _load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return _parse_yaml(handle.read())


def _load_yaml_resource(resource: Traversable) -> dict[str, Any]:
    with resource.open("r", encoding="utf-8") as handle:
        return _parse_yaml(handle.read())


def _parse_yaml(text: str) -> dict[str, Any]:
    parsed = yaml.safe_load(text) or {}
    if not isinstance(parsed, dict):  # pragma: no cover - safeguard
        msg = "Configuration must be a mapping at the top level"
        raise ValueError(msg)
    return parsed
