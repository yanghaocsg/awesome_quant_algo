# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LeapQuant: per-window recurrent-state quantization for linear attention."""

from .config import LeapQuantConfig
from .state import (
    EncodedState,
    LeapQuantSpec,
    closes_window,
    decode,
    encode,
    fit_compensator_tokens,
    is_window_start,
    roundtrip,
    state_mse,
)

__all__ = [
    "EncodedState",
    "LeapQuantConfig",
    "LeapQuantSpec",
    "closes_window",
    "decode",
    "encode",
    "fit_compensator_tokens",
    "is_window_start",
    "roundtrip",
    "state_mse",
]
