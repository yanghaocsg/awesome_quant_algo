# SPDX-License-Identifier: Apache-2.0
"""Pure-torch port of the official TurboQuant algorithm (0xsero/turboquant, arXiv:2504.19874).

This is a faithful, self-contained port of the upstream core:
  * lloyd_max  : data-OBLIVIOUS analytic Lloyd-Max codebook for the distribution
                 that a unit vector's coordinates follow after a Haar rotation
                 (Beta, Gaussian N(0,1/d) approximation for d >= 64), solved with
                 scipy.integrate.quad conditional expectations -- NOT data k-means.
  * quantizer  : Haar random orthogonal rotation + per-coordinate scalar quantization
                 (TurboQuantMSE), and the two-stage QJL variant (TurboQuantProd).

Only dependencies: torch + scipy (both preinstalled in Colab). CPU and CUDA supported:
the codebook / random matrices are built on CPU (float32) and moved to the tensor device.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from scipy import integrate
from scipy.stats import beta as beta_dist
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Distribution PDFs (rotated unit-vector coordinates)
# ---------------------------------------------------------------------------
def _beta_pdf(x: float, d: int) -> float:
    """Beta((d-1)/2,(d-1)/2) PDF on [-1/sqrt(d), 1/sqrt(d)] for a rotated coordinate."""
    alpha = (d - 1) / 2.0
    scale = 1.0 / math.sqrt(d)
    if abs(x) >= scale:
        return 0.0
    t = (x / scale + 1.0) / 2.0
    return beta_dist.pdf(t, alpha, alpha) / (2.0 * scale)


def _gaussian_pdf(x: float, d: int) -> float:
    """Gaussian N(0,1/d) approximation, accurate for d >= 64."""
    sigma = 1.0 / math.sqrt(d)
    return float(norm.pdf(x, loc=0.0, scale=sigma))


# ---------------------------------------------------------------------------
# Analytic (data-oblivious) Lloyd-Max codebook
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def solve_lloyd_max(
    d: int,
    bits: int,
    *,
    use_exact: bool = False,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (centroids[2^bits], boundaries[2^bits-1]) for the known rotated-coordinate PDF.

    Lloyd-Max conditions: boundaries are midpoints of adjacent centroids; centroids are
    conditional expectations E[X | partition], evaluated analytically via scipy quad.
    The result depends only on (d, bits) -- no input data is used.
    """
    n_levels = 1 << bits
    pdf = _beta_pdf if use_exact else _gaussian_pdf

    sigma = 1.0 / math.sqrt(d)
    lo, hi = -3.0 * sigma, 3.0 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo] + boundaries + [hi]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            if b - a < 1e-15:
                new_centroids.append((a + b) / 2.0)
                continue
            numer, _ = integrate.quad(lambda x: x * pdf(x, d), a, b)
            denom, _ = integrate.quad(lambda x: pdf(x, d), a, b)
            if denom < 1e-15:
                new_centroids.append((a + b) / 2.0)
            else:
                new_centroids.append(numer / denom)

        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries_final = [
        (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
    ]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries_final, dtype=torch.float32),
    )


@dataclass
class LloydMaxCodebook:
    """Precomputed data-oblivious codebook. quantize via bucketize, dequantize via lookup."""

    centroids: torch.Tensor
    boundaries: torch.Tensor
    bits: int
    dim: int

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        bounds = self.boundaries.to(x.device)
        return torch.bucketize(x, bounds)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        cents = self.centroids.to(indices.device)
        return cents[indices]


# ---------------------------------------------------------------------------
# Haar random orthogonal rotation
# ---------------------------------------------------------------------------
def generate_rotation_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    """Haar-distributed orthogonal matrix via QR of a Gaussian (CPU float32), sign-corrected."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(
        dim, dim, generator=gen, device="cpu", dtype=torch.float32
    )
    q, r = torch.linalg.qr(gaussian)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    return q * diag_sign.unsqueeze(0)


# ---------------------------------------------------------------------------
# Stage 1: rotation + Lloyd-Max MSE quantizer
# ---------------------------------------------------------------------------
class TurboQuantMSE:
    """Normalize (store norm) -> Haar rotate -> data-oblivious per-coordinate Lloyd-Max."""

    def __init__(self, dim: int, bits: int, *, seed: int = 42) -> None:
        self.dim = dim
        self.bits = bits
        centroids, boundaries = solve_lloyd_max(dim, bits)
        self.codebook = LloydMaxCodebook(centroids, boundaries, bits, dim)
        self.rotation = generate_rotation_matrix(dim, seed=seed)

    def quantize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Input last dimension {x.shape[-1]} does not match quantizer dim {self.dim}"
            )
        orig_shape = x.shape
        flat = x.reshape(-1, self.dim).float()

        norms = torch.norm(flat, dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-10)

        pi = self.rotation.to(flat.device)
        rotated = normalized @ pi.T
        indices = self.codebook.quantize(rotated)
        return indices.reshape(orig_shape), norms.reshape(*orig_shape[:-1], 1)

    def dequantize(
        self, indices: torch.Tensor, norms: torch.Tensor
    ) -> torch.Tensor:
        if indices.shape[-1] != self.dim:
            raise ValueError(
                f"Indices last dimension {indices.shape[-1]} does not match quantizer dim {self.dim}"
            )
        orig_shape = indices.shape
        flat_idx = indices.reshape(-1, self.dim)
        flat_norms = norms.reshape(-1, 1)

        reconstructed = self.codebook.dequantize(flat_idx)
        pi = self.rotation.to(reconstructed.device)
        unrotated = reconstructed @ pi
        result = unrotated * flat_norms
        return result.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Stage 2: (bits-1) MSE + 1-bit QJL residual sign correction (unbiased inner product)
# ---------------------------------------------------------------------------
class TurboQuantProd:
    """Two-stage quantizer producing unbiased inner-product estimates (QJL correction)."""

    def __init__(
        self,
        dim: int,
        bits: int,
        *,
        qjl_dim: int | None = None,
        seed: int = 42,
    ) -> None:
        if bits < 2:
            raise ValueError(f"TurboQuantProd requires bits >= 2, got {bits}")
        self.dim = dim
        self.bits = bits
        self.mse_quantizer = TurboQuantMSE(dim, bits - 1, seed=seed)

        self.qjl_dim = qjl_dim or dim
        gen = torch.Generator(device="cpu").manual_seed(seed + 1)
        self.qjl_matrix = (
            torch.randn(self.qjl_dim, dim, generator=gen, device="cpu")
            / math.sqrt(self.qjl_dim)
        )

    def quantize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices, norms = self.mse_quantizer.quantize(x)
        reconstructed = self.mse_quantizer.dequantize(indices, norms)
        residual = x.float() - reconstructed
        residual_norms = torch.norm(residual, dim=-1, keepdim=True)

        s = self.qjl_matrix.to(x.device)
        projected = residual.reshape(-1, self.dim) @ s.T
        qjl_signs = torch.sign(projected).reshape(*x.shape[:-1], self.qjl_dim)
        qjl_signs[qjl_signs == 0] = 1.0
        return indices, norms, qjl_signs, residual_norms

    def dequantize(
        self,
        indices: torch.Tensor,
        norms: torch.Tensor,
        qjl_signs: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        return self.mse_quantizer.dequantize(indices, norms)

    def estimate_inner_product(
        self,
        query: torch.Tensor,
        indices: torch.Tensor,
        norms: torch.Tensor,
        qjl_signs: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        if query.shape[-1] != self.dim:
            raise ValueError(
                f"Query last dimension {query.shape[-1]} does not match quantizer dim {self.dim}"
            )
        k_mse = self.mse_quantizer.dequantize(indices, norms)
        mse_term = (query.float() * k_mse).sum(dim=-1, keepdim=True)

        s = self.qjl_matrix.to(query.device)
        q_projected = query.float().reshape(-1, self.dim) @ s.T
        q_projected = q_projected.reshape(*query.shape[:-1], self.qjl_dim)
        qjl_correction = (q_projected * qjl_signs).sum(dim=-1, keepdim=True)
        scale = residual_norms * math.sqrt(math.pi / 2.0) / self.qjl_dim
        qjl_term = scale * qjl_correction
        return mse_term + qjl_term


# ---------------------------------------------------------------------------
# Self-contained compressed KV cache (nibble packing + compression statistics)
# ---------------------------------------------------------------------------
class CompressedKVCache:
    """Pure-torch compressed KV cache over [batch, heads, seq, head_dim] tensors.

    Each layer is compressed independently (no cross-layer recurrence). 4-bit uses
    nibble packing identical to the upstream implementation; other widths store uint8
    indices plus fp32 norms. This is a storage/accuracy model: add() compresses, get()
    decompresses, and storage_bytes()/compression_stats() report the real memory saving.
    """

    def __init__(self, head_dim: int, bits: int, *, seed: int = 42) -> None:
        if bits == 4 and head_dim % 2 != 0:
            raise ValueError("nibble packing requires an even head_dim")
        self.head_dim = head_dim
        self.bits = bits
        self.nibble_packed = bits == 4
        self.quantizer = TurboQuantMSE(head_dim, bits, seed=seed)
        self._stored: dict[int, tuple[torch.Tensor, torch.Tensor, tuple]] = {}

    @staticmethod
    def _nibble_pack(indices: torch.Tensor) -> torch.Tensor:
        even = indices[..., 0::2]
        odd = indices[..., 1::2]
        return (even << 4) | odd

    @staticmethod
    def _nibble_unpack(packed: torch.Tensor) -> torch.Tensor:
        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        return torch.stack([high, low], dim=-1).flatten(-2)

    def add(self, x: torch.Tensor, layer_idx: int) -> None:
        if x.shape[-1] != self.head_dim:
            raise ValueError("tensor last dim does not match head_dim")
        indices, norms = self.quantizer.quantize(x)
        norms = norms.reshape(*x.shape[:-1]).float()
        if self.nibble_packed:
            stored = self._nibble_pack(indices.to(torch.uint8))
        else:
            stored = indices.to(torch.uint8)
        self._stored[layer_idx] = (stored, norms, x.shape)

    def get(self, layer_idx: int) -> torch.Tensor:
        stored, norms, shape = self._stored[layer_idx]
        if self.nibble_packed:
            indices = self._nibble_unpack(stored)
        else:
            indices = stored.long()
        flat_norms = norms.reshape(-1, 1)
        reconstructed = self.quantizer.dequantize(
            indices.reshape(-1, self.head_dim), flat_norms
        )
        return reconstructed.reshape(shape)

    def storage_bytes(self) -> int:
        total = 0
        for stored, norms, _ in self._stored.values():
            total += stored.nelement() * stored.element_size()
            total += norms.nelement() * norms.element_size()
        return total

    def baseline_bytes(self) -> int:
        """FP16 storage that the uncompressed cache would use."""
        total = 0
        for _, _, shape in self._stored.values():
            total += int(torch.tensor(shape).prod().item()) * 2
        return total

    def compression_stats(self) -> dict:
        compressed = self.storage_bytes()
        baseline = self.baseline_bytes()
        return {
            "num_layers": len(self._stored),
            "compression_ratio": baseline / compressed,
            "compressed_mib": compressed / (1024 ** 2),
            "baseline_mib": baseline / (1024 ** 2),
            "savings_mib": (baseline - compressed) / (1024 ** 2),
        }
