# SPDX-License-Identifier: Apache-2.0
"""TurboQuant pure-torch regression tests.

Thresholds and the synthetic dataset are aligned with the official turboquant-vllm-npu
test suite (tests/conftest.py, test_lloyd_max.py, test_quantizer.py, test_kv_cache_regression.py,
test_per_layer_cosine.py). Runs on CPU; the same code path also executes on CUDA in Colab.

Usage:
    pytest test_turboquant_regression.py -v
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import turboquant_torch as tq

# Same constants as the official conftest.
DIM = 128
BITS = 3
SEED = 42
N_SAMPLES = 500
N_PAIRS = 300


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Codebook properties
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bits", [2, 3, 4, 5])
def test_codebook_counts(bits):
    centroids, boundaries = tq.solve_lloyd_max(DIM, bits)
    assert len(centroids) == 2 ** bits
    assert len(boundaries) == 2 ** bits - 1


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_codebook_sorted_and_symmetric(bits):
    centroids, boundaries = tq.solve_lloyd_max(DIM, bits)
    assert torch.all(centroids[1:] > centroids[:-1])
    assert torch.all(boundaries[1:] > boundaries[:-1])
    assert abs(centroids.mean().item()) < 1e-4


def test_beta_pdf_support():
    assert tq._beta_pdf(1.0, 16) == 0.0
    assert tq._beta_pdf(-1.0, 16) == 0.0
    assert tq._beta_pdf(0.0, 16) > 0.0


# ---------------------------------------------------------------------------
# MSE quantizer: theoretical bound, norm preservation, zero vector, ordering
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_mse_quantization_bound(bits):
    quant = tq.TurboQuantMSE(DIM, bits, seed=SEED)
    x = torch.randn(N_SAMPLES, DIM)
    x = x / x.norm(dim=-1, keepdim=True)
    indices, norms = quant.quantize(x)
    assert indices.min() >= 0 and indices.max() < 2 ** bits
    reconstructed = quant.dequantize(indices, norms)
    mse = (x - reconstructed).pow(2).mean().item()
    bound = 5 * math.sqrt(3) * math.pi / 2 * (1 / 4 ** bits)
    assert mse < bound


def test_mse_improves_with_bits():
    mses = []
    for bits in (2, 3, 4):
        quant = tq.TurboQuantMSE(DIM, bits, seed=SEED)
        x = torch.randn(N_SAMPLES, DIM)
        x = x / x.norm(dim=-1, keepdim=True)
        idx, n = quant.quantize(x)
        mses.append((x - quant.dequantize(idx, n)).pow(2).mean().item())
    assert mses[0] > mses[1] > mses[2]


def test_norm_preserved():
    quant = tq.TurboQuantMSE(DIM, BITS, seed=SEED)
    x = torch.randn(200, DIM)
    original = x.norm(dim=-1)
    _, stored = quant.quantize(x)
    rel = ((original - stored.squeeze()) / (original + 1e-10)).abs().mean().item()
    assert rel < 0.01


def test_zero_vector():
    quant = tq.TurboQuantMSE(DIM, BITS, seed=SEED)
    z = torch.zeros(8, DIM)
    idx, n = quant.quantize(z)
    assert torch.all(quant.dequantize(idx, n).abs() < 1e-6)


def test_large_magnitude_vectors():
    quant = tq.TurboQuantMSE(DIM, BITS, seed=SEED)
    x = torch.randn(50, DIM) * 1000.0
    idx, n = quant.quantize(x)
    cos = F.cosine_similarity(x, quant.dequantize(idx, n), dim=-1).mean().item()
    # Official threshold: magnitude is stored separately, 3-bit empirical cosine ~0.86.
    assert cos > 0.80


# ---------------------------------------------------------------------------
# QJL two-stage: unbiased inner product
# ---------------------------------------------------------------------------
def test_prod_inner_product():
    prod = tq.TurboQuantProd(DIM, BITS, seed=SEED)
    query = torch.randn(N_PAIRS, DIM) * 0.1
    key = torch.randn(N_PAIRS, DIM) * 0.1
    true_ip = (query * key).sum(-1)
    idx, n, signs, rn = prod.quantize(key)
    est_ip = prod.estimate_inner_product(query, idx, n, signs, rn).squeeze(-1)
    rel_bias = (est_ip - true_ip).mean().abs().item() / (true_ip.abs().mean() + 1e-10)
    corr = torch.corrcoef(torch.stack([true_ip, est_ip]))[0, 1].item()
    assert rel_bias < 0.15
    assert corr > 0.9


# ---------------------------------------------------------------------------
# KV cache: storage-format lossless, quality thresholds, compression stats
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("bits", "min_cos"), [(3, 0.90), (4, 0.95)])
def test_kv_cache_long_sequence(bits, min_cos):
    num_layers, num_heads, seq = 36, 8, 1024 + 32
    kc = tq.CompressedKVCache(DIM, bits, seed=SEED)
    vc = tq.CompressedKVCache(DIM, bits, seed=SEED)
    storage_cos, quant_cos = [], []
    for layer in range(num_layers):
        for cache in (kc, vc):
            x = torch.randn(1, num_heads, seq, DIM)
            idx0, n0 = cache.quantizer.quantize(x)
            baseline = cache.quantizer.dequantize(idx0, n0)
            cache.add(x, layer)
            packed = cache.get(layer)
            assert torch.isfinite(packed).all()
            storage_cos.append(F.cosine_similarity(
                baseline.flatten(), packed.flatten(), dim=0).item())
            quant_cos.append(F.cosine_similarity(
                x.flatten(), packed.flatten(), dim=0).item())
    assert all(c > 0.999 for c in storage_cos)
    assert all(c > min_cos for c in quant_cos)

    storage = kc.storage_bytes() + vc.storage_bytes()
    baseline_bytes = kc.baseline_bytes() + vc.baseline_bytes()
    ratio = baseline_bytes / storage
    savings_mib = (baseline_bytes - storage) / 1024 ** 2
    assert ratio > 1.9
    assert savings_mib > 50
