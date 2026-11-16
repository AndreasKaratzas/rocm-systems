# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Resolve ML tracing backends and operator patterns without analysis imports."""

from typing import Any, Dict, List, Optional, Tuple

from ml_api_tracing.constants import (
    KNOWN_ML_API_BACKENDS,
    OPERATOR_BACKENDS,
    USER_REGION_BACKEND,
)

OPERATOR_LABELS = {"torch": "PyTorch", "triton": "Triton", USER_REGION_BACKEND: "User"}


def operator_backend_labels(backends: List[str]) -> str:
    """Join backend display names, using ML API for an unscoped view."""
    if not backends:
        return "ML API"
    return ", ".join(OPERATOR_LABELS.get(backend, backend) for backend in backends)


def resolve_profile_selection(raw: Optional[List[str]]) -> List[str]:
    """Return the ordered, unique backends selected by --trace-ml-api.

    None disables tracing. A bare flag selects every known ML API backend.
    User ROCTx regions are included in each trace and need no explicit backend.
    """
    if raw is None:
        return []
    if not raw:
        return list(KNOWN_ML_API_BACKENDS)
    for backend in raw:
        if backend not in KNOWN_ML_API_BACKENDS:
            raise ValueError(
                f"Unknown --trace-ml-api value {backend!r}. "
                f"Values: {', '.join(KNOWN_ML_API_BACKENDS)}"
            )
    return list(dict.fromkeys(raw))


def traced_frameworks_from_config(profiling_config: Dict[str, Any]) -> List[str]:
    """Return the frameworks that --trace-ml-api traced, from profiling_config.yaml."""
    return list(profiling_config.get("trace_ml_api") or [])


def check_backends_traced(backends: List[str], traced_frameworks: List[str]) -> None:
    """Raise ValueError unless the workload traced every given backend.

    User regions are recorded by every --trace-ml-api run.
    """
    if not traced_frameworks:
        raise ValueError(
            "Workload was not profiled with --trace-ml-api. Profile again with "
            "--trace-ml-api to use --list-operators or --operator."
        )
    for backend in backends:
        if backend in KNOWN_ML_API_BACKENDS and backend not in traced_frameworks:
            raise ValueError(
                f"Workload was not profiled with --trace-ml-api {backend}."
            )


def split_backend_prefix(pattern: str) -> Tuple[Optional[str], str]:
    """Split 'torch:*conv*' into ('torch', '*conv*').

    A pattern without a prefix returns (None, pattern). The '::' in names such
    as aten::addmm is not a prefix.
    """
    prefix, separator, rest = pattern.partition(":")
    if not separator or rest.startswith(":"):
        return None, pattern
    return prefix, rest or "**"


def bind_operator_patterns(
    patterns: List[str], traced_frameworks: List[str]
) -> Dict[str, List[str]]:
    """Group --operator patterns by the backend they match in one workload.

    A pattern without a prefix uses the prefix before it. Before any prefix, it
    matches every traced framework and user regions. An empty pattern list
    matches every operator. Raises ValueError for an unknown prefix or an
    untraced framework.

    For example, ['torch:*conv*', '*relu*', 'triton:*matmul*'] selects Torch
    conv and relu operators, followed by Triton matmul operators.
    """
    patterns_by_backend: Dict[str, List[str]] = {}
    backends = [*traced_frameworks, USER_REGION_BACKEND]
    for token in patterns or ["**"]:
        prefix, pattern = split_backend_prefix(token)
        if prefix is not None:
            if prefix not in OPERATOR_BACKENDS:
                raise ValueError(
                    f"Unknown --operator prefix {prefix!r}. "
                    f"Prefixes: {', '.join(OPERATOR_BACKENDS)}"
                )
            backends = [prefix]
        for backend in backends:
            patterns_by_backend.setdefault(backend, []).append(pattern)
    check_backends_traced(list(patterns_by_backend), traced_frameworks)
    return patterns_by_backend
