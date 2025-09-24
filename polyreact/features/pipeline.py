"""Feature pipeline construction utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..config import Config, DescriptorSettings, FeatureBackendSettings
from .descriptors import DescriptorConfig, DescriptorFeaturizer
from .plm import PLMEmbedder


@dataclass(slots=True)
class FeaturePipelineState:
    backend_type: str
    descriptor_featurizer: DescriptorFeaturizer | None
    plm_scaler: StandardScaler | None
    descriptor_config: DescriptorConfig | None
    plm_model_name: str | None
    plm_layer_pool: str | None
    cache_dir: str | None
    device: str
    feature_names: list[str] = field(default_factory=list)


class FeaturePipeline:
    """Fit/transform feature matrices according to configuration."""

    def __init__(
        self,
        *,
        backend: FeatureBackendSettings,
        descriptors: DescriptorSettings,
        device: str,
        cache_dir_override: str | None = None,
        plm_model_override: str | None = None,
        layer_pool_override: str | None = None,
    ) -> None:
        self.backend = backend
        self.descriptor_settings = descriptors
        self.device = device
        self.cache_dir_override = cache_dir_override
        self.plm_model_override = plm_model_override
        self.layer_pool_override = layer_pool_override

        self._descriptor: DescriptorFeaturizer | None = None
        self._plm: PLMEmbedder | None = None
        self._plm_scaler: StandardScaler | None = None
        self._feature_names: list[str] = []

    def fit_transform(self, df, *, heavy_only: bool, batch_size: int = 8) -> np.ndarray:  # noqa: ANN001
        backend_type = self.backend.type if self.backend.type else "descriptors"
        self._validate_heavy_support(backend_type, heavy_only)
        sequences = _extract_sequences(df, heavy_only=heavy_only)

        if backend_type == "descriptors":
            self._descriptor = _build_descriptor_featurizer(self.descriptor_settings)
            features = self._descriptor.fit_transform(sequences)
            self._feature_names = list(self._descriptor.feature_names_ or [])
            self._plm = None
            self._plm_scaler = None
            return features.astype(np.float32)

        if backend_type == "plm":
            self._descriptor = None
            self._plm = _build_plm_embedder(
                self.backend,
                device=self.device,
                cache_dir_override=self.cache_dir_override,
                plm_model_override=self.plm_model_override,
                layer_pool_override=self.layer_pool_override,
            )
            embeddings = self._plm.embed(sequences, batch_size=batch_size)
            if self.backend.standardize:
                self._plm_scaler = StandardScaler()
                embeddings = self._plm_scaler.fit_transform(embeddings)
            else:
                self._plm_scaler = None
            self._feature_names = [f"plm_{i}" for i in range(embeddings.shape[1])]
            return embeddings.astype(np.float32)

        if backend_type == "concat":
            descriptor = _build_descriptor_featurizer(self.descriptor_settings)
            desc_features = descriptor.fit_transform(sequences)
            plm = _build_plm_embedder(
                self.backend,
                device=self.device,
                cache_dir_override=self.cache_dir_override,
                plm_model_override=self.plm_model_override,
                layer_pool_override=self.layer_pool_override,
            )
            embeddings = plm.embed(sequences, batch_size=batch_size)
            if self.backend.standardize:
                plm_scaler = StandardScaler()
                embeddings = plm_scaler.fit_transform(embeddings)
            else:
                plm_scaler = None
            self._descriptor = descriptor
            self._plm = plm
            self._plm_scaler = plm_scaler
            self._feature_names = list(descriptor.feature_names_ or []) + [
                f"plm_{i}" for i in range(embeddings.shape[1])
            ]
            return np.concatenate([desc_features, embeddings], axis=1).astype(np.float32)

        msg = f"Unsupported feature backend: {backend_type}"
        raise ValueError(msg)

    def fit(self, df, *, heavy_only: bool, batch_size: int = 8) -> "FeaturePipeline":  # noqa: ANN001
        backend_type = self.backend.type if self.backend.type else "descriptors"
        self._validate_heavy_support(backend_type, heavy_only)
        sequences = _extract_sequences(df, heavy_only=heavy_only)

        if backend_type == "descriptors":
            self._descriptor = _build_descriptor_featurizer(self.descriptor_settings)
            self._descriptor.fit(sequences)
            self._feature_names = list(self._descriptor.feature_names_ or [])
            self._plm = None
            self._plm_scaler = None
        elif backend_type == "plm":
            self._descriptor = None
            self._plm = _build_plm_embedder(
                self.backend,
                device=self.device,
                cache_dir_override=self.cache_dir_override,
                plm_model_override=self.plm_model_override,
                layer_pool_override=self.layer_pool_override,
            )
            embeddings = self._plm.embed(sequences, batch_size=batch_size)
            if self.backend.standardize:
                self._plm_scaler = StandardScaler()
                embeddings = self._plm_scaler.fit_transform(embeddings)
            else:
                self._plm_scaler = None
            self._feature_names = [f"plm_{i}" for i in range(embeddings.shape[1])]
        elif backend_type == "concat":
            descriptor = _build_descriptor_featurizer(self.descriptor_settings)
            desc_features = descriptor.fit_transform(sequences)
            plm = _build_plm_embedder(
                self.backend,
                device=self.device,
                cache_dir_override=self.cache_dir_override,
                plm_model_override=self.plm_model_override,
                layer_pool_override=self.layer_pool_override,
            )
            embeddings = plm.embed(sequences, batch_size=batch_size)
            if self.backend.standardize:
                plm_scaler = StandardScaler()
                embeddings = plm_scaler.fit_transform(embeddings)
            else:
                plm_scaler = None
            self._descriptor = descriptor
            self._plm = plm
            self._plm_scaler = plm_scaler
            self._feature_names = list(descriptor.feature_names_ or []) + [
                f"plm_{i}" for i in range(embeddings.shape[1])
            ]
        else:  # pragma: no cover - defensive branch
            msg = f"Unsupported feature backend: {backend_type}"
            raise ValueError(msg)
        return self

    def transform(self, df, *, heavy_only: bool, batch_size: int = 8) -> np.ndarray:  # noqa: ANN001
        backend_type = self.backend.type if self.backend.type else "descriptors"
        self._validate_heavy_support(backend_type, heavy_only)
        sequences = _extract_sequences(df, heavy_only=heavy_only)

        if backend_type == "descriptors":
            if self._descriptor is None:
                msg = "Descriptor featurizer is not fitted"
                raise RuntimeError(msg)
            features = self._descriptor.transform(sequences)
        elif backend_type == "plm":
            if self._plm is None:
                msg = "PLM embedder is not initialised"
                raise RuntimeError(msg)
            embeddings = self._plm.embed(sequences, batch_size=batch_size)
            if self.backend.standardize and self._plm_scaler is not None:
                embeddings = self._plm_scaler.transform(embeddings)
            features = embeddings
        elif backend_type == "concat":
            if self._descriptor is None or self._plm is None:
                msg = "Feature pipeline not fitted"
                raise RuntimeError(msg)
            desc_features = self._descriptor.transform(sequences)
            embeddings = self._plm.embed(sequences, batch_size=batch_size)
            if self.backend.standardize and self._plm_scaler is not None:
                embeddings = self._plm_scaler.transform(embeddings)
            features = np.concatenate([desc_features, embeddings], axis=1)
        else:  # pragma: no cover - defensive branch
            msg = f"Unsupported feature backend: {backend_type}"
            raise ValueError(msg)

        return features.astype(np.float32)

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def get_state(self) -> FeaturePipelineState:
        descriptor = self._descriptor
        if descriptor is not None and descriptor.numberer is not None:
            if hasattr(descriptor.numberer, "_runner"):
                descriptor.numberer._runner = None  # type: ignore[attr-defined]
        return FeaturePipelineState(
            backend_type=self.backend.type,
            descriptor_featurizer=descriptor,
            plm_scaler=self._plm_scaler,
            descriptor_config=_build_descriptor_config(self.descriptor_settings),
            plm_model_name=self._effective_plm_model_name,
            plm_layer_pool=self._effective_layer_pool,
            cache_dir=self._effective_cache_dir,
            device=self.device,
            feature_names=self._feature_names,
        )

    def load_state(self, state: FeaturePipelineState) -> None:
        self.backend.type = state.backend_type
        if state.plm_model_name:
            self.backend.plm_model_name = state.plm_model_name
            self.plm_model_override = state.plm_model_name
        if state.plm_layer_pool:
            self.backend.layer_pool = state.plm_layer_pool
            self.layer_pool_override = state.plm_layer_pool
        if state.cache_dir:
            self.backend.cache_dir = state.cache_dir
            self.cache_dir_override = state.cache_dir
        if state.descriptor_config:
            self.descriptor_settings = DescriptorSettings(
                use_anarci=state.descriptor_config.use_anarci,
                regions=tuple(state.descriptor_config.regions),
                features=tuple(state.descriptor_config.features),
                ph=state.descriptor_config.ph,
            )
        self._descriptor = state.descriptor_featurizer
        self._plm_scaler = state.plm_scaler
        self._feature_names = state.feature_names
        if self.backend.type in {"plm", "concat"}:
            self._plm = _build_plm_embedder(
                self.backend,
                device=self.device,
                cache_dir_override=self.backend.cache_dir,
                plm_model_override=self.backend.plm_model_name,
                layer_pool_override=self.backend.layer_pool,
            )
        else:
            self._plm = None

    @property
    def _effective_plm_model_name(self) -> str | None:
        if self.backend.type not in {"plm", "concat"}:
            return None
        return self.plm_model_override or self.backend.plm_model_name

    @property
    def _effective_layer_pool(self) -> str | None:
        if self.backend.type not in {"plm", "concat"}:
            return None
        return self.layer_pool_override or self.backend.layer_pool

    @property
    def _effective_cache_dir(self) -> str | None:
        if self.backend.type not in {"plm", "concat"}:
            return None
        if self.cache_dir_override is not None:
            return self.cache_dir_override
        return self.backend.cache_dir

    def _validate_heavy_support(self, backend_type: str, heavy_only: bool) -> None:
        if heavy_only:
            return
        if backend_type == "descriptors" and self.descriptor_settings.use_anarci:
            msg = "Descriptor backend with ANARCI currently supports heavy-chain only inference."
            raise ValueError(msg)
        if backend_type == "concat" and self.descriptor_settings.use_anarci:
            msg = "Concat backend with descriptors requires heavy-chain only data."
            raise ValueError(msg)


def build_feature_pipeline(
    config: Config,
    *,
    backend_override: str | None = None,
    plm_model_override: str | None = None,
    cache_dir_override: str | None = None,
    layer_pool_override: str | None = None,
) -> FeaturePipeline:
    backend = FeatureBackendSettings(**asdict(config.feature_backend))
    if backend_override:
        backend.type = backend_override
    pipeline = FeaturePipeline(
        backend=backend,
        descriptors=config.descriptors,
        device=config.device,
        cache_dir_override=cache_dir_override,
        plm_model_override=plm_model_override,
        layer_pool_override=layer_pool_override,
    )
    return pipeline


def _build_descriptor_featurizer(settings: DescriptorSettings) -> DescriptorFeaturizer:
    descriptor_config = _build_descriptor_config(settings)
    return DescriptorFeaturizer(config=descriptor_config, standardize=True)


def _build_descriptor_config(settings: DescriptorSettings) -> DescriptorConfig:
    return DescriptorConfig(
        use_anarci=settings.use_anarci,
        regions=tuple(settings.regions),
        features=tuple(settings.features),
        ph=settings.ph,
    )


def _build_plm_embedder(
    backend: FeatureBackendSettings,
    *,
    device: str,
    cache_dir_override: str | None,
    plm_model_override: str | None,
    layer_pool_override: str | None,
) -> PLMEmbedder:
    model_name = plm_model_override or backend.plm_model_name
    cache_dir = cache_dir_override or backend.cache_dir
    layer_pool = layer_pool_override or backend.layer_pool
    return PLMEmbedder(
        model_name=model_name,
        layer_pool=layer_pool,
        device=device,
        cache_dir=cache_dir,
    )


def _extract_sequences(df, heavy_only: bool) -> Sequence[str]:  # noqa: ANN001
    if heavy_only or "light_seq" not in df.columns:
        return df["heavy_seq"].fillna("").astype(str).tolist()
    heavy = df["heavy_seq"].fillna("").astype(str)
    light = df["light_seq"].fillna("").astype(str)
    return (heavy + "|" + light).tolist()
