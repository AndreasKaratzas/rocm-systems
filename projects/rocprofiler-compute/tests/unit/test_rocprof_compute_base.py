# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_base.py."""

import argparse
import re
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import common
import pandas as pd
import pytest

from rocprof_compute_base import RocProfCompute

MODULE = "rocprof_compute_base"
# Profile selection fields that _validate_profile_mode_arguments reads
PROFILE_SELECTION = {
    "filter_blocks": [],
    "set_selected": None,
    "bench_only": False,
    "no_roof": False,
}
# List option fields that _validate_list_option_exclusions reads
LIST_OPTIONS = {
    "debug_list_metrics_gfx_id": None,
    "list_metrics": None,
    "list_blocks": None,
    "list_available_metrics": False,
    "filter_blocks": [],
}


FAILING_OPERATOR_OPTIONS = [
    pytest.param(["triton"], None, id="untraced-listing"),
    pytest.param(None, ["triton:*matmul*"], id="untraced-filter"),
    pytest.param(None, ["invalid-prefix:*"], id="invalid-filter"),
]


def build_args(
    *,
    mode="analyze",
    list_stats=False,
    output_format="stdout",
    list_operators=None,
    operator=None,
) -> argparse.Namespace:
    """Build a Namespace with the fields handle_analyze_args reads."""
    return argparse.Namespace(
        mode=mode,
        list_stats=list_stats,
        output_format=output_format,
        list_operators=list_operators,
        operator=operator,
    )


def rocprof_compute_with_args(args: argparse.Namespace) -> RocProfCompute:
    """Construct a RocProfCompute without invoking __init__."""
    instance = RocProfCompute.__new__(RocProfCompute)
    instance._RocProfCompute__args = args
    instance._RocProfCompute__mode = args.mode
    return instance


def write_analysis_lifecycle_workload(parent, name, gpu_arch):
    workload = parent / name / "GPU"
    workload.mkdir(parents=True)
    (workload / "profiling_config.yaml").write_text(
        "trace_ml_api:\n- torch\n", encoding="utf-8"
    )
    pd.DataFrame({"SQ_WAVES": [1]}).to_csv(
        workload / "results_0.csv.gz", index=False, compression="gzip"
    )
    if gpu_arch is not None:
        pd.DataFrame([{"gpu_arch": gpu_arch}]).to_csv(
            workload / "sysinfo.csv", index=False
        )
    return workload


def create_analysis_lifecycle_soc(_args, machine_specs):
    return SimpleNamespace(_mspec=machine_specs)


@pytest.fixture
def analysis_lifecycle_soc_modules(monkeypatch):
    """Keep main's real SoC import while supplying one available test architecture."""
    module_name = "rocprof_compute_soc.soc_lifecycle_available"
    available = ModuleType(module_name)
    available.lifecycle_available_soc = create_analysis_lifecycle_soc
    monkeypatch.setitem(sys.modules, module_name, available)
    monkeypatch.setattr(
        f"{MODULE}.generate_machine_specs",
        lambda _args, sysinfo: SimpleNamespace(gpu_arch=sysinfo["gpu_arch"]),
    )


@pytest.mark.parametrize(("listing", "operator"), FAILING_OPERATOR_OPTIONS)
@pytest.mark.parametrize("reverse_paths", [False, True], ids=["first", "later"])
@pytest.mark.parametrize("failure", ["sysinfo", "soc"])
def test_run_analysis_initialization_error_precedes_operator_authorization(
    tmp_path,
    monkeypatch,
    analysis_lifecycle_soc_modules,
    listing,
    operator,
    reverse_paths,
    failure,
):
    """Main finishes sysinfo/SoC initialization before any selector is authorized."""
    bad_workload = write_analysis_lifecycle_workload(
        tmp_path,
        "bad",
        None if failure == "sysinfo" else "lifecycle_missing",
    )
    good_workload = write_analysis_lifecycle_workload(
        tmp_path, "good", "lifecycle_available"
    )
    paths = [bad_workload, good_workload]
    if reverse_paths:
        paths.reverse()
    args = build_args(list_operators=listing, operator=operator)
    args.path = [[str(path)] for path in paths]
    args.output_name = None
    instance = rocprof_compute_with_args(args)
    instance._RocProfCompute__analyze_mode = "cli"
    instance._RocProfCompute__supported_archs = {}
    instance._RocProfCompute__soc = {}
    monkeypatch.setattr(instance, "print_graphic", lambda: None)
    exception = FileNotFoundError if failure == "sysinfo" else ModuleNotFoundError
    expected_message = (
        str(bad_workload / "sysinfo.csv")
        if failure == "sysinfo"
        else "rocprof_compute_soc.soc_lifecycle_missing"
    )

    with pytest.raises(exception, match=re.escape(expected_message)):
        instance.run_analysis()

    assert args.list_operators == listing
    assert args.operator == operator
    assert set(instance._RocProfCompute__soc) == (
        {"lifecycle_available"} if reverse_paths else set()
    )


@pytest.mark.parametrize(
    "output_format, expect_error",
    [("stdout", False), ("txt", False), ("csv", True), ("db", True)],
)
def test_handle_analyze_args_list_stats_output_format(output_format, expect_error):
    """--list-stats is rejected with csv and db output."""
    args = build_args(list_stats=True, output_format=output_format)
    instance = rocprof_compute_with_args(args)
    if expect_error:
        with pytest.raises(SystemExit):
            instance.handle_analyze_args()
    else:
        instance.handle_analyze_args()


def test_handle_analyze_args_listing_ignores_operator_filter(monkeypatch):
    """--list-operators wins, and a warning says --operator is ignored."""
    warning = common.patch_console(monkeypatch, MODULE, "warning")["warning"]
    args = build_args(list_operators=["torch"], operator=["torch:*relu*"])
    rocprof_compute_with_args(args).handle_analyze_args()
    assert "--operator is ignored" in warning.call_args.args[1]


@pytest.mark.parametrize(
    "operator_flags",
    [{"list_operators": ["torch"]}, {"operator": ["torch:*relu*"]}],
)
@pytest.mark.parametrize(
    "output_format, expect_error",
    [("stdout", False), ("txt", True), ("csv", True), ("db", True)],
)
def test_handle_analyze_args_operator_flags_require_stdout(
    operator_flags, output_format, expect_error, monkeypatch
):
    """--list-operators and --operator are rejected unless output is stdout."""
    error = common.patch_console(
        monkeypatch, MODULE, "error", error=Mock(side_effect=SystemExit(1))
    )["error"]
    args = build_args(output_format=output_format, **operator_flags)
    instance = rocprof_compute_with_args(args)
    if expect_error:
        with pytest.raises(SystemExit):
            instance.handle_analyze_args()
        assert "only supported with --output-format stdout" in error.call_args.args[1]
    else:
        instance.handle_analyze_args()


@pytest.mark.parametrize(
    "selection, expect_error",
    [
        ({"filter_blocks": ["2", "4"]}, False),
        ({"set_selected": "launch_stats"}, False),
        ({"bench_only": True}, False),
        ({"filter_blocks": ["2"], "set_selected": "launch_stats"}, True),
        ({"filter_blocks": ["4"], "bench_only": True}, True),
        ({"set_selected": "launch_stats", "bench_only": True}, True),
        ({"bench_only": True, "no_roof": True}, True),
    ],
    ids=[
        "blocks",
        "set",
        "bench_only",
        "blocks_with_set",
        "blocks_with_bench_only",
        "set_with_bench_only",
        "bench_only_with_no_roof",
    ],
)
def test_validate_profile_mode_arguments(selection, expect_error):
    """Blocks, --set and --roofline-bench-only exclude each other."""
    args = argparse.Namespace(mode="profile", **{**PROFILE_SELECTION, **selection})
    instance = rocprof_compute_with_args(args)
    if expect_error:
        with pytest.raises(SystemExit):
            instance._validate_profile_mode_arguments()
    else:
        instance._validate_profile_mode_arguments()


@pytest.mark.parametrize(
    "options, named_option",
    [
        ({"debug_list_metrics_gfx_id": "gfx90a"}, "with --list-metrics"),
        ({"list_metrics": ["sol"], "filter_blocks": ["2"]}, "--list-metrics"),
        ({"list_blocks": "gfx950", "filter_blocks": ["2"]}, "--list-blocks"),
        (
            {"list_available_metrics": True, "filter_blocks": ["2"]},
            "--list-available-metrics",
        ),
    ],
    ids=[
        "gfx_id_requires_list_metrics",
        "list_metrics_with_blocks",
        "list_blocks_with_blocks",
        "list_available_metrics_with_blocks",
    ],
)
def test_validate_list_option_exclusions(monkeypatch, options, named_option):
    """Each rejected combination names the list option in the error."""
    error = common.patch_console(
        monkeypatch, MODULE, "error", error=Mock(side_effect=SystemExit(1))
    )["error"]
    args = argparse.Namespace(mode=None, **{**LIST_OPTIONS, **options})
    with pytest.raises(SystemExit):
        rocprof_compute_with_args(args)._validate_list_option_exclusions()
    assert named_option in error.call_args.args[0]
