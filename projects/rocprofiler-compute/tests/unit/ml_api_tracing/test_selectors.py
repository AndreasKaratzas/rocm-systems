# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for independent ML API selector policy."""

from typing import Dict, List, Optional, get_type_hints

import pytest

from ml_api_tracing.constants import KNOWN_ML_API_BACKENDS
from ml_api_tracing.selectors import (
    bind_operator_patterns,
    check_backends_traced,
    operator_backend_labels,
    resolve_profile_selection,
    split_backend_prefix,
    traced_frameworks_from_config,
)


@pytest.mark.parametrize(
    "profiling_config, expected",
    [
        ({}, []),
        ({"trace_ml_api": None}, []),
        ({"trace_ml_api": ["torch", "triton"]}, ["torch", "triton"]),
    ],
)
def test_traced_frameworks_from_config(profiling_config, expected):
    assert traced_frameworks_from_config(profiling_config) == expected


def test_check_backends_traced_accepts_user_regions():
    check_backends_traced(["torch", "user"], ["torch"])


@pytest.mark.parametrize(
    "backends, traced_frameworks, message",
    [
        pytest.param(["triton"], ["torch"], "--trace-ml-api triton", id="untraced"),
        pytest.param([], [], "Profile again with --trace-ml-api", id="no_trace"),
    ],
)
def test_check_backends_traced_rejects(backends, traced_frameworks, message):
    with pytest.raises(ValueError, match=message):
        check_backends_traced(backends, traced_frameworks)


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("torch:*conv*", ("torch", "*conv*")),
        ("torch:*/aten::addmm", ("torch", "*/aten::addmm")),
        ("*/aten::addmm", (None, "*/aten::addmm")),
        ("*relu*", (None, "*relu*")),
        ("user:", ("user", "**")),
    ],
)
def test_split_backend_prefix(pattern, expected):
    assert split_backend_prefix(pattern) == expected


@pytest.mark.parametrize(
    "patterns, traced_frameworks, expected",
    [
        pytest.param(
            ["torch:*conv*", "*relu*", "triton:*matmul*"],
            ["torch", "triton"],
            {"torch": ["*conv*", "*relu*"], "triton": ["*matmul*"]},
            id="prefix_carries_forward",
        ),
        pytest.param(
            ["*relu*"],
            ["torch"],
            {"torch": ["*relu*"], "user": ["*relu*"]},
            id="no_prefix_matches_traced_and_user",
        ),
        pytest.param(
            [],
            ["triton"],
            {"triton": ["**"], "user": ["**"]},
            id="no_patterns_match_everything",
        ),
        pytest.param(
            ["user:*UserDefined*"],
            ["torch"],
            {"user": ["*UserDefined*"]},
            id="user_prefix",
        ),
    ],
)
def test_bind_operator_patterns(patterns, traced_frameworks, expected):
    assert bind_operator_patterns(patterns, traced_frameworks) == expected


@pytest.mark.parametrize(
    "patterns, traced_frameworks, message",
    [
        pytest.param(["toch:*relu*"], ["torch"], "prefix 'toch'", id="unknown"),
        pytest.param(["triton:*mm*"], ["torch"], "api triton", id="untraced"),
        pytest.param(["*relu*"], [], "Profile again", id="no_trace"),
    ],
)
def test_bind_operator_patterns_rejects(patterns, traced_frameworks, message):
    with pytest.raises(ValueError, match=message):
        bind_operator_patterns(patterns, traced_frameworks)


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param(None, [], id="absent"),
        pytest.param([], list(KNOWN_ML_API_BACKENDS), id="bare"),
        pytest.param(
            ["triton", "torch", "triton"],
            ["triton", "torch"],
            id="ordered-unique",
        ),
    ],
)
def test_profile_selection_preserves_order_without_mutating_input(raw, expected):
    original = None if raw is None else raw.copy()
    assert resolve_profile_selection(raw) == expected
    assert raw == original


@pytest.mark.parametrize("backend", ["user", "torch,triton", "unknown"])
def test_profile_selection_rejects_unknown_backend(backend):
    with pytest.raises(ValueError, match="Unknown --trace-ml-api value"):
        resolve_profile_selection([backend])


def test_operator_comma_joined_token_is_one_glob():
    assert bind_operator_patterns(
        ["torch:*conv*,*relu*,triton:*matmul*"], ["torch"]
    ) == {"torch": ["*conv*,*relu*,triton:*matmul*"]}


def test_binding_one_workload_keeps_raw_patterns_for_another():
    raw = ["*relu*"]
    torch_bound = bind_operator_patterns(raw, ["torch"])
    triton_bound = bind_operator_patterns(raw, ["triton"])
    assert torch_bound == {"torch": ["*relu*"], "user": ["*relu*"]}
    assert triton_bound == {"triton": ["*relu*"], "user": ["*relu*"]}
    torch_bound["user"].append("*torch_only*")
    assert triton_bound["user"] == ["*relu*"]
    assert raw == ["*relu*"]


@pytest.mark.parametrize(
    "backends, expected",
    [
        pytest.param([], "ML API", id="unscoped"),
        pytest.param(["torch"], "PyTorch", id="torch"),
        pytest.param(["triton"], "Triton", id="triton"),
        pytest.param(["user"], "User", id="user"),
        pytest.param(
            ["user", "triton", "torch"],
            "User, Triton, PyTorch",
            id="ordered-frameworks-and-user",
        ),
        pytest.param(["custom"], "custom", id="unknown-display-name"),
    ],
)
def test_operator_backend_labels_keep_plain_display_names(backends, expected):
    original = backends.copy()
    assert operator_backend_labels(backends) == expected
    assert backends == original


def test_public_selector_annotations_are_runtime_evaluable():
    assert get_type_hints(resolve_profile_selection) == {
        "raw": Optional[List[str]],
        "return": List[str],
    }
    assert get_type_hints(bind_operator_patterns) == {
        "patterns": List[str],
        "traced_frameworks": List[str],
        "return": Dict[str, List[str]],
    }


def test_selectors_import_fresh_with_only_stdlib_under_profile_guard(
    assert_stdlib_import,
):
    assert_stdlib_import("ml_api_tracing.selectors")
