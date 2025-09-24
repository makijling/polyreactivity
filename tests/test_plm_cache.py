from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from polyreact.features.plm import PLMEmbedder


class _DummyTokenizer:
    def __call__(
        self,
        sequences,
        *,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        add_special_tokens: bool = True,
        return_special_tokens_mask: bool = False,
    ):
        if isinstance(sequences, str):
            sequences = [sequences]
        batch = len(sequences)
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        tensor_len = max_len + (2 if add_special_tokens else 0)
        input_ids = torch.zeros((batch, tensor_len), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        special_tokens_mask = torch.zeros_like(input_ids)
        for i, seq in enumerate(sequences):
            start = 1 if add_special_tokens else 0
            if add_special_tokens:
                special_tokens_mask[i, 0] = 1
            for j, _aa in enumerate(seq):
                input_ids[i, start + j] = j + 1
                attention_mask[i, start + j] = 1
            if add_special_tokens:
                special_tokens_mask[i, start + len(seq)] = 1
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_special_tokens_mask:
            output["special_tokens_mask"] = special_tokens_mask
        return output


class _DummyModel:
    def __init__(self) -> None:
        self.call_count = 0

    def eval(self) -> "_DummyModel":
        return self

    def to(self, device: str) -> "_DummyModel":
        return self

    def __call__(self, *, input_ids, attention_mask):  # noqa: ANN001
        self.call_count += 1
        batch_size, seq_len = input_ids.shape
        hidden = torch.arange(batch_size * seq_len * 4, dtype=torch.float32)
        hidden = hidden.reshape(batch_size, seq_len, 4)
        return SimpleNamespace(last_hidden_state=hidden)


def _dummy_loader(model_name: str, device: str):  # noqa: D401
    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    return tokenizer, model


@pytest.fixture()
def embedder(tmp_path: Path) -> PLMEmbedder:
    return PLMEmbedder(
        model_name="dummy",
        layer_pool="mean",
        device="cpu",
        cache_dir=tmp_path,
        model_loader=_dummy_loader,
    )


def test_plm_embedder_caches_embeddings(embedder: PLMEmbedder, tmp_path: Path) -> None:
    sequences = ["ACDE", "WXYZ", "ACDE"]
    first = embedder.embed(sequences, batch_size=2)
    second = embedder.embed(sequences, batch_size=2)

    assert np.allclose(first, second)
    assert embedder.model.call_count == 1

    model_dir = tmp_path / "dummy"
    cache_files = list(model_dir.glob("*.npy"))
    assert len(cache_files) == 2


def test_plm_embedder_layer_pool_cls(embedder: PLMEmbedder) -> None:
    embedder.layer_pool = "cls"
    sequences = ["ACDE"]
    output = embedder.embed(sequences, batch_size=1)
    assert output.shape == (1, 4)
