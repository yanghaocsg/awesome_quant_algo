# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup configuration and constraint checks for Ascend LeapQuant.

Kept separate from ``vllm_ascend.ascend_config`` on purpose: M1 is an
experimental, opt-in path and must not widen the core ``AscendConfig`` surface
or the upstream ``MambaDType`` literal. Everything is read from the free-form
``additional_config`` dict, mirroring how ``enable_sparse_sfa_c8`` and
``xlite_graph_config`` are consumed today.
"""

import logging
from dataclasses import dataclass
from typing import Any

from .state import LeapQuantSpec

logger = logging.getLogger(__name__)

__all__ = ["LeapQuantConfig", "LeapQuantSpec"]

_DEFAULT_SPEC = LeapQuantSpec()


def _strict_bool(value: Any, path: str) -> bool:
    """Accept only real JSON booleans.

    ``validate_additional_config_bool`` in ``vllm_ascend.ascend_config`` makes
    the same distinction; we duplicate the 4 lines instead of importing that
    module so this package stays importable without vLLM / NPU installed.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a JSON boolean, got {type(value).__name__}")
    return value


def _strict_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be a JSON integer, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class LeapQuantConfig:
    """Resolved LeapQuant settings for one engine."""

    enabled: bool
    spec: LeapQuantSpec

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "LeapQuantConfig":
        additional = getattr(vllm_config, "additional_config", None) or {}
        raw = additional.get("leapquant")
        if raw is None:
            return cls(enabled=False, spec=_DEFAULT_SPEC)
        if not isinstance(raw, dict):
            raise ValueError(f"additional_config.leapquant must be an object, got {type(raw).__name__}")

        if not _strict_bool(raw.get("enabled", False), "additional_config.leapquant.enabled"):
            return cls(enabled=False, spec=_DEFAULT_SPEC)

        spec = LeapQuantSpec(
            window=_strict_int(raw.get("window", 16), "additional_config.leapquant.window"),
            bits=_strict_int(raw.get("bits", 8), "additional_config.leapquant.bits"),
            compensator_tokens=_strict_int(
                raw.get("compensator_tokens", 4),
                "additional_config.leapquant.compensator_tokens",
            ),
            smooth=_strict_bool(raw.get("smooth", True), "additional_config.leapquant.smooth"),
            power_iters=_strict_int(raw.get("power_iters", 4), "additional_config.leapquant.power_iters"),
        )
        config = cls(enabled=True, spec=spec)
        config.validate(vllm_config)
        return config

    def validate(self, vllm_config: Any) -> None:
        """Reject combinations whose window boundary M1 cannot keep aligned.

        These are correctness constraints, not performance ones: silently
        degrading them would make the measured accuracy unrelated to the paper.
        """
        cache_config = getattr(vllm_config, "cache_config", None)
        mamba_cache_mode = getattr(cache_config, "mamba_cache_mode", "none")
        if mamba_cache_mode == "align":
            # align mode checkpoints the state every block_size tokens, and
            # block_size is not guaranteed to be a multiple of the window.
            raise ValueError(
                "LeapQuant does not support Mamba prefix caching yet "
                "(cache_config.mamba_cache_mode='align'); "
                "please pass --no-enable-prefix-caching."
            )
        if getattr(cache_config, "enable_prefix_caching", False):
            logger.warning(
                "LeapQuant: prefix caching is enabled but mamba_cache_mode is not "
                "'align'; state checkpoints will not be re-quantized at window "
                "boundaries."
            )

        if getattr(vllm_config, "speculative_config", None) is not None:
            # num_accepted_tokens can advance a request by more than one token
            # per step, so a window may be skipped entirely.
            raise ValueError("LeapQuant does not support speculative decoding yet.")

        if getattr(vllm_config, "kv_transfer_config", None) is not None:
            # PD-disaggregated state transfer assumes the fp32/bf16 layout.
            raise ValueError("LeapQuant does not support KV transfer (PD disaggregation) yet.")

        model_config = getattr(vllm_config, "model_config", None)
        dtype = getattr(model_config, "dtype", None)
        if dtype is not None and str(dtype) not in ("torch.bfloat16", "bfloat16", "torch.float32", "float32"):
            raise ValueError(f"LeapQuant requires a bf16 or fp32 model dtype, got {dtype}.")

        logger.info(
            "LeapQuant enabled: window=%d, bits=%d, compensator_tokens=%d, smooth=%s",
            self.spec.window,
            self.spec.bits,
            self.spec.compensator_tokens,
            self.spec.smooth,
        )
