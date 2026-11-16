# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for backend-selection constants."""

from typing import Tuple, get_type_hints

from ml_api_tracing import constants


def test_known_ml_api_backends_keeps_ordered_immutable_names():
    assert constants.KNOWN_ML_API_BACKENDS == ("torch", "triton")
    assert isinstance(constants.KNOWN_ML_API_BACKENDS, tuple)
    assert get_type_hints(constants)["KNOWN_ML_API_BACKENDS"] == Tuple[str, ...]


def test_user_region_backend_keeps_marker_identity_separate_from_frameworks():
    assert constants.USER_REGION_BACKEND == "user"
    assert constants.USER_REGION_BACKEND not in constants.KNOWN_ML_API_BACKENDS
    assert get_type_hints(constants)["USER_REGION_BACKEND"] is str


def test_operator_backends_include_user_after_traced_frameworks():
    assert constants.OPERATOR_BACKENDS == ("torch", "triton", "user")
    assert get_type_hints(constants)["OPERATOR_BACKENDS"] == Tuple[str, ...]


def test_constants_import_fresh_with_only_stdlib_under_profile_guard(
    assert_stdlib_import,
):
    assert_stdlib_import("ml_api_tracing.constants")
