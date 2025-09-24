"""Protein language model embeddings backend with caching support."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoModel = None
    AutoTokenizer = None

try:  # pragma: no cover - optional dependency
    import esm
except ImportError:  # pragma: no cover - optional dependency
    esm = None

from .anarsi import AnarciNumberer

ModelLoader = Callable[[str, str], Tuple[object, nn.Module]]

if esm is not None:  # pragma: no cover - optional dependency
    _ESM1V_LOADERS = {
        "esm1v_t33_650m_ur90s_1": esm.pretrained.esm1v_t33_650M_UR90S_1,
        "esm1v_t33_650m_ur90s_2": esm.pretrained.esm1v_t33_650M_UR90S_2,
        "esm1v_t33_650m_ur90s_3": esm.pretrained.esm1v_t33_650M_UR90S_3,
        "esm1v_t33_650m_ur90s_4": esm.pretrained.esm1v_t33_650M_UR90S_4,
        "esm1v_t33_650m_ur90s_5": esm.pretrained.esm1v_t33_650M_UR90S_5,
    }
else:  # pragma: no cover - optional dependency
    _ESM1V_LOADERS: dict[str, Callable[[], tuple[nn.Module, object]]] = {}


class _ESMTokenizer:
    """Callable wrapper that mimics Hugging Face tokenizers for ESM models."""

    def __init__(self, alphabet) -> None:  # noqa: ANN001
        self.alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()

    def __call__(
        self,
        sequences: Sequence[str],
        *,
        return_tensors: str = "pt",
        padding: bool = True,  # noqa: FBT002
        truncation: bool = True,  # noqa: FBT002
        add_special_tokens: bool = True,  # noqa: FBT002
        return_special_tokens_mask: bool = True,  # noqa: FBT002
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":  # pragma: no cover - defensive branch
            msg = "ESM tokenizer only supports return_tensors='pt'"
            raise ValueError(msg)
        data = [(str(idx), (seq or "").upper()) for idx, seq in enumerate(sequences)]
        _labels, _strings, tokens = self._batch_converter(data)
        attention_mask = (tokens != self.alphabet.padding_idx).long()
        special_tokens = torch.zeros_like(tokens)
        specials = {
            self.alphabet.padding_idx,
            self.alphabet.cls_idx,
            self.alphabet.eos_idx,
        }
        for special in specials:
            special_tokens |= tokens == special
        output: dict[str, torch.Tensor] = {
            "input_ids": tokens,
            "attention_mask": attention_mask,
        }
        if return_special_tokens_mask:
            output["special_tokens_mask"] = special_tokens.long()
        return output


class _ESMModelWrapper(nn.Module):
    """Adapter providing a Hugging Face style interface for ESM models."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.layer_index = getattr(model, "num_layers", None)
        if self.layer_index is None:
            msg = "Unable to determine final layer for ESM model"
            raise AttributeError(msg)

    def eval(self) -> "_ESMModelWrapper":  # pragma: no cover - trivial
        self.model.eval()
        return self

    def to(self, device: str) -> "_ESMModelWrapper":  # pragma: no cover - trivial
        self.model.to(device)
        return self

    def forward(self, input_ids: torch.Tensor, **_):  # noqa: ANN003
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                repr_layers=[self.layer_index],
                return_contacts=False,
            )
        hidden = outputs["representations"][self.layer_index]
        return SimpleNamespace(last_hidden_state=hidden)

    __call__ = forward


@dataclass(slots=True)
class PLMConfig:
    model_name: str = "facebook/esm1v_t33_650M_UR90S_1"
    layer_pool: str = "mean"
    cache_dir: Path = Path(".cache/embeddings")
    device: str = "auto"


class PLMEmbedder:
    """Embed amino-acid sequences using a transformer model with caching."""

    def __init__(
        self,
        model_name: str = "facebook/esm1v_t33_650M_UR90S_1",
        *,
        layer_pool: str = "mean",
        device: str = "auto",
        cache_dir: str | Path | None = None,
        numberer: AnarciNumberer | None = None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self.model_name = model_name
        self.layer_pool = layer_pool
        self.device = self._resolve_device(device)
        self.cache_dir = Path(cache_dir or ".cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.numberer = numberer
        self.model_loader = model_loader
        self._tokenizer: object | None = None
        self._model: nn.Module | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @property
    def tokenizer(self):  # noqa: D401
        if self._tokenizer is None:
            tokenizer, model = self._load_model_components()
            self._tokenizer = tokenizer
            self._model = model
        return self._tokenizer

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            tokenizer, model = self._load_model_components()
            self._tokenizer = tokenizer
            self._model = model
        return self._model

    def _load_model_components(self) -> Tuple[object, nn.Module]:
        if self.model_loader is not None:
            tokenizer, model = self.model_loader(self.model_name, self.device)
            return tokenizer, model

        if self._is_esm1v_model(self.model_name):
            return self._load_esm_model()

        if AutoModel is None or AutoTokenizer is None:  # pragma: no cover - optional dependency
            msg = "transformers must be installed to use PLMEmbedder"
            raise ImportError(msg)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        model.eval()
        model.to(self.device)
        return tokenizer, model

    def _load_esm_model(self) -> Tuple[object, nn.Module]:
        if esm is None:  # pragma: no cover - optional dependency
            msg = (
                "The 'esm' package is required to use ESM-1v models."
            )
            raise ImportError(msg)

        normalized = self._canonical_esm_name(self.model_name)
        loader = _ESM1V_LOADERS.get(normalized)
        if loader is None:  # pragma: no cover - guard branch
            msg = f"Unsupported ESM-1v model: {self.model_name}"
            raise ValueError(msg)

        model, alphabet = loader()
        model.eval()
        model.to(self.device)
        tokenizer = _ESMTokenizer(alphabet)
        wrapper = _ESMModelWrapper(model)
        return tokenizer, wrapper

    @staticmethod
    def _canonical_esm_name(model_name: str) -> str:
        name = model_name.lower()
        if "/" in name:
            name = name.split("/")[-1]
        return name

    @classmethod
    def _is_esm1v_model(cls, model_name: str) -> bool:
        return cls._canonical_esm_name(model_name).startswith("esm1v")

    def embed(self, sequences: Iterable[str], *, batch_size: int = 8) -> np.ndarray:
        batch_sequences = list(sequences)
        if not batch_sequences:
            return np.empty((0, 0), dtype=np.float32)

        outputs: List[np.ndarray | None] = [None] * len(batch_sequences)
        unique_to_compute: dict[str, List[Tuple[int, Path]]] = {}
        model_dir = self.cache_dir / self._normalized_model_name()
        model_dir.mkdir(parents=True, exist_ok=True)

        cache_hits: list[tuple[int, Path]] = []
        for idx, sequence in enumerate(batch_sequences):
            cache_path = self._sequence_cache_path(model_dir, sequence)
            if cache_path.exists():
                cache_hits.append((idx, cache_path))
            else:
                unique_to_compute.setdefault(sequence, []).append((idx, cache_path))

        if cache_hits:
            loaders = [path for _, path in cache_hits]
            max_workers = min(len(loaders), 32)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for (idx, _), embedding in zip(cache_hits, executor.map(np.load, loaders), strict=True):
                    outputs[idx] = embedding

        if unique_to_compute:
            embeddings = self._compute_embeddings(list(unique_to_compute.keys()), batch_size=batch_size)
            for sequence, embedding in zip(unique_to_compute.keys(), embeddings, strict=True):
                targets = unique_to_compute[sequence]
                for idx, cache_path in targets:
                    outputs[idx] = embedding
                    np.save(cache_path, embedding)
        if any(item is None for item in outputs):  # pragma: no cover - safety
            msg = "Failed to compute embeddings for all sequences"
            raise RuntimeError(msg)
        array_outputs = [np.asarray(item, dtype=np.float32) for item in outputs]  # type: ignore[arg-type]
        return np.stack(array_outputs, axis=0)

    def _compute_embeddings(self, sequences: Sequence[str], *, batch_size: int) -> List[np.ndarray]:
        tokenizer = self.tokenizer
        model = self.model
        model.eval()
        embeddings: List[np.ndarray] = []
        for start in range(0, len(sequences), batch_size):
            chunk = list(sequences[start : start + batch_size])
            tokenized = self._tokenize(tokenizer, chunk)
            model_inputs: dict[str, torch.Tensor] = {}
            aux_inputs: dict[str, torch.Tensor] = {}
            for key, value in tokenized.items():
                if isinstance(value, torch.Tensor):
                    tensor_value = value.to(self.device)
                else:
                    tensor_value = value
                if key == "special_tokens_mask":
                    aux_inputs[key] = tensor_value
                else:
                    model_inputs[key] = tensor_value
            with torch.no_grad():
                outputs = model(**model_inputs)
                hidden_states = outputs.last_hidden_state.detach().cpu()
            attention_mask = model_inputs.get("attention_mask")
            special_tokens_mask = aux_inputs.get("special_tokens_mask")
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.detach().cpu()
            if isinstance(special_tokens_mask, torch.Tensor):
                special_tokens_mask = special_tokens_mask.detach().cpu()

            for idx, sequence in enumerate(chunk):
                hidden = hidden_states[idx]
                mask = attention_mask[idx] if isinstance(attention_mask, torch.Tensor) else None
                special_mask = (
                    special_tokens_mask[idx]
                    if isinstance(special_tokens_mask, torch.Tensor)
                    else None
                )
                embedding = self._pool_hidden(hidden, mask, special_mask, sequence)
                embeddings.append(embedding)
        return embeddings

    def _tokenize(self, tokenizer, sequences: Sequence[str]):
        if hasattr(tokenizer, "__call__"):
            return tokenizer(
                list(sequences),
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
                return_special_tokens_mask=True,
            )
        msg = "Tokenizer does not implement __call__"
        raise TypeError(msg)

    def _pool_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        special_mask: torch.Tensor | None,
        sequence: str,
    ) -> np.ndarray:
        if attention_mask is None:
            attention = torch.ones(hidden.size(0), dtype=torch.float32)
        else:
            attention = attention_mask.to(dtype=torch.float32)
        if special_mask is not None:
            attention = attention * (1.0 - special_mask.to(dtype=torch.float32))
        if attention.sum() == 0:
            attention = torch.ones_like(attention)

        if self.layer_pool == "mean":
            return self._masked_mean(hidden, attention)
        if self.layer_pool == "cls":
            return hidden[0].detach().cpu().numpy()
        if self.layer_pool == "per_token_mean_cdrh3":
            return self._pool_cdrh3(hidden, attention, sequence)
        msg = f"Unsupported layer pool: {self.layer_pool}"
        raise ValueError(msg)

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        weights = mask.unsqueeze(-1)
        weighted = hidden * weights
        denom = weights.sum()
        if denom == 0:
            pooled = hidden.mean(dim=0)
        else:
            pooled = weighted.sum(dim=0) / denom
        return pooled.detach().cpu().numpy()

    def _pool_cdrh3(self, hidden: torch.Tensor, mask: torch.Tensor, sequence: str) -> np.ndarray:
        numberer = self.numberer
        if numberer is None:
            numberer = AnarciNumberer()
            self.numberer = numberer
        numbered = numberer.number_sequence(sequence)
        cdr = numbered.regions.get("CDRH3", "")
        if not cdr:
            return self._masked_mean(hidden, mask)
        sequence_upper = sequence.upper()
        start = sequence_upper.find(cdr.upper())
        if start == -1:
            return self._masked_mean(hidden, mask)
        residues_idx = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        if not residues_idx:
            return self._masked_mean(hidden, mask)
        end = start + len(cdr)
        if end > len(residues_idx):
            return self._masked_mean(hidden, mask)
        cdr_token_positions = residues_idx[start:end]
        if not cdr_token_positions:
            return self._masked_mean(hidden, mask)
        cdr_mask = torch.zeros_like(mask)
        for pos in cdr_token_positions:
            cdr_mask[pos] = 1.0
        return self._masked_mean(hidden, cdr_mask)

    def _sequence_cache_path(self, model_dir: Path, sequence: str) -> Path:
        digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()
        return model_dir / f"{digest}.npy"

    def _normalized_model_name(self) -> str:
        if self._is_esm1v_model(self.model_name):
            return self._canonical_esm_name(self.model_name)
        return self.model_name.replace("/", "_")
