# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/analysis_base.py."""

import argparse
import logging
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import common
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base

MODULE = "rocprof_compute_analyze.analysis_base"

# An empty path list leaves pre_processing() nothing to walk but the sink setup.
PRE_PROCESSING_ARGS = {
    "path": [],
    "gpu_kernel": None,
    "gpu_id": None,
    "gpu_dispatch_id": None,
}

FAILING_OPERATOR_OPTIONS = [
    pytest.param(["triton"], None, id="untraced-listing"),
    pytest.param(None, ["triton:*matmul*"], id="untraced-filter"),
    pytest.param(None, ["invalid-prefix:*"], id="invalid-filter"),
]


def write_traced_workload(parent: Path, workload_name: str, backend: str) -> Path:
    """Create a workload with profiling metadata for one ML backend."""
    workload = parent / workload_name / "MI350"
    workload.mkdir(parents=True)
    (workload / "profiling_config.yaml").write_text(
        f"trace_ml_api:\n- {backend}\n", encoding="utf-8"
    )
    return workload


def record_error_and_exit(errors, *arguments, **_kwargs):
    """Record the console error and preserve its fatal behavior."""
    errors.append(arguments)
    raise SystemExit(1)


def mixed_path_orders(torch_workload: Path, triton_workload: Path):
    return (
        [[str(torch_workload)], [str(triton_workload)]],
        [[str(triton_workload)], [str(torch_workload)]],
    )


def write_counter_results(workload: Path) -> Path:
    """Write the readable counter data required by workload validation."""
    results_path = workload / "results_0.csv.gz"
    pd.DataFrame({"SQ_WAVES": [1]}).to_csv(results_path, index=False)
    return results_path


def test_sanitize_rejects_paths_sharing_a_workload_name(tmp_path, monkeypatch) -> None:
    """Reject two paths whose last two components match."""
    mock_error = common.patch_console(monkeypatch, MODULE, "error")["error"]
    paths = [[str(tmp_path / parent / "vcopy" / "MI300")] for parent in ("a", "b")]
    for path in paths:
        Path(path[0]).mkdir(parents=True)

    # The mock records instead of exiting, so sanitize runs on to a later error.
    with pytest.raises(SystemExit):
        OmniAnalyze_Base(argparse.Namespace(path=paths), {}).sanitize()

    assert "last two components" in mock_error.call_args.args[1]


@pytest.mark.parametrize(("listing", "operator"), FAILING_OPERATOR_OPTIONS)
@pytest.mark.parametrize("invalid_workload_index", [0, 1], ids=["first", "later"])
@pytest.mark.parametrize("workload_error", ["missing", "empty"])
def test_sanitize_workload_error_precedes_operator_error(
    tmp_path, caplog, listing, operator, invalid_workload_index, workload_error
) -> None:
    """All workloads must have counter data before selectors are authorized."""
    workloads = [
        write_traced_workload(tmp_path, "first", "torch"),
        write_traced_workload(tmp_path, "later", "torch"),
    ]
    invalid_workload = workloads[invalid_workload_index]
    for workload in workloads:
        if workload != invalid_workload:
            write_counter_results(workload)
    expected_error = "[analysis] No profiling data found."
    if workload_error == "empty":
        results_path = invalid_workload / "results_0.csv.gz"
        pd.DataFrame().to_csv(results_path, index=False)
        expected_error = (
            f"[profiling] No counter data in {results_path}.\n"
            "Profiling data could be corrupt."
        )
    args = argparse.Namespace(
        path=[[str(workload)] for workload in workloads],
        list_operators=listing,
        operator=operator,
        output_name=None,
    )
    analyzer = OmniAnalyze_Base(args, {})

    with pytest.raises(SystemExit) as error:
        analyzer.sanitize()

    assert error.value.code == 1
    assert caplog.messages[-1] == expected_error
    assert analyzer._operator_patterns_by_path == {}
    assert args.list_operators == listing
    assert args.operator == operator


@pytest.mark.parametrize(("listing", "operator"), FAILING_OPERATOR_OPTIONS)
@pytest.mark.parametrize(
    ("output_name", "output_format", "existing"),
    [
        pytest.param("bad/name", "txt", False, id="invalid-name"),
        pytest.param("analysis_report", "txt", True, id="existing-txt"),
        pytest.param("analysis_report", "db", True, id="existing-db"),
        pytest.param("analysis_report", "csv", True, id="existing-folder"),
    ],
)
def test_sanitize_output_error_precedes_operator_error(
    tmp_path,
    monkeypatch,
    caplog,
    listing,
    operator,
    output_name,
    output_format,
    existing,
) -> None:
    """Invalid or occupied report destinations take precedence over selectors."""
    workload = write_traced_workload(tmp_path, "app", "torch")
    write_counter_results(workload)
    monkeypatch.chdir(tmp_path)
    if existing:
        destination_name = output_name
        if output_format in ("txt", "db"):
            destination_name += f".{output_format}"
        destination = tmp_path / destination_name
        if output_format == "csv":
            destination.mkdir()
        else:
            destination.write_text("existing report\n", encoding="utf-8")
        expected_error = (
            f"Analysis output file/folder {destination_name} already exists. "
            "Please choose a different name."
        )
    else:
        expected_error = (
            "[analysis] Analysis output file/folder name must "
            "contain only alphanumeric characters "
            "or underscores (_), hyphens (-)."
        )
    args = argparse.Namespace(
        path=[[str(workload)]],
        list_operators=listing,
        operator=operator,
        output_name=output_name,
        output_format=output_format,
    )
    analyzer = OmniAnalyze_Base(args, {})

    with pytest.raises(SystemExit) as error:
        analyzer.sanitize()

    assert error.value.code == 1
    assert caplog.messages[-1] == expected_error
    assert analyzer._operator_patterns_by_path == {}
    if existing and output_format != "csv":
        assert destination.read_text(encoding="utf-8") == "existing report\n"


@pytest.mark.parametrize(("listing", "operator"), FAILING_OPERATOR_OPTIONS)
def test_bind_operator_selectors_keeps_sanitize_iteration_notice(
    tmp_path, caplog, listing, operator
) -> None:
    """The generic iteration notice remains visible when a selector is invalid."""
    workload = write_traced_workload(tmp_path, "app", "torch")
    config_path = workload / "profiling_config.yaml"
    config_path.write_text(
        "trace_ml_api:\n- torch\niteration_multiplexing: median\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        path=[[str(workload)]],
        list_operators=listing,
        operator=operator,
        output_name=None,
    )
    caplog.set_level(logging.INFO)

    analyzer = OmniAnalyze_Base(args, {})
    analyzer.sanitize()
    with pytest.raises(SystemExit) as error:
        analyzer.bind_operator_selectors(str(workload))

    assert error.value.code == 1
    assert caplog.messages[0] == (
        "[analysis] Profiling data was collected using iteration multiplexing.\n\t"
        "Metrics are calculated based on partially available counter data."
    )
    assert caplog.messages[-1].startswith(f"[ml api trace] {workload}:")


@pytest.mark.parametrize(
    "generic_config",
    [
        pytest.param("iteration_multiplexing: median\n", id="iteration"),
        pytest.param("filter_blocks:\n- pc_sampling\n", id="pc-sampling-only"),
    ],
)
def test_sanitize_first_config_controls_generic_workload_validation(
    tmp_path, generic_config
) -> None:
    """Generic validation uses the first config while selectors bind per workload."""
    torch_workload = write_traced_workload(tmp_path, "first", "torch")
    triton_workload = write_traced_workload(tmp_path, "later", "triton")
    (torch_workload / "profiling_config.yaml").write_text(
        f"trace_ml_api:\n- torch\n{generic_config}", encoding="utf-8"
    )
    args = argparse.Namespace(
        path=[[str(torch_workload)], [str(triton_workload)]],
        list_operators=None,
        operator=[],
        output_name=None,
    )
    analyzer = OmniAnalyze_Base(args, {})

    analyzer.sanitize()

    assert analyzer.get_profiling_config()["trace_ml_api"] == ["torch"]
    assert analyzer._operator_patterns_by_path == {}
    if not analyzer.pc_sampling_only():
        for workload in (torch_workload, triton_workload):
            analyzer.bind_operator_selectors(str(workload))
        assert analyzer._operator_patterns_by_path == {
            str(torch_workload): {"torch": ["**"], "user": ["**"]},
            str(triton_workload): {"triton": ["**"], "user": ["**"]},
        }
    assert args.operator == []


# ---------------------------------------------------------------------------
# pre_processing output_format dispatch
# ---------------------------------------------------------------------------


def test_pre_processing_txt_creates_named_file(tmp_path, monkeypatch) -> None:
    """--output-format txt with --output-name writes <name>.txt in the cwd."""
    mocks = common.patch_console(monkeypatch, MODULE, "debug", "log", "warning")
    monkeypatch.setattr(OmniAnalyze_Base, "initalize_runs", lambda self: {})
    monkeypatch.chdir(tmp_path)

    analyzer = OmniAnalyze_Base(
        argparse.Namespace(
            output_format="txt", output_name="analysis_report", **PRE_PROCESSING_ARGS
        ),
        {},
    )
    analyzer.pre_processing()

    try:
        report = tmp_path / "analysis_report.txt"
        assert report.is_file()
        assert not analyzer._output.closed
        assert Path(analyzer._output.name).resolve() == report
        assert analyzer._output.writable()
        assert "analysis_report.txt" in mocks["warning"].call_args.args[1]
    finally:
        analyzer._output.close()


def test_pre_processing_txt_default_name_is_uuid(tmp_path, monkeypatch) -> None:
    """Without --output-name the txt file falls back to rocprof_compute_<uuid>."""
    common.patch_console(monkeypatch, MODULE, "debug", "log", "warning")
    monkeypatch.setattr(OmniAnalyze_Base, "initalize_runs", lambda self: {})
    monkeypatch.chdir(tmp_path)

    analyzer = OmniAnalyze_Base(
        argparse.Namespace(
            output_format="txt", output_name=None, **PRE_PROCESSING_ARGS
        ),
        {},
    )
    analyzer.pre_processing()

    try:
        created = list(tmp_path.iterdir())
        assert len(created) == 1
        assert created[0].match("rocprof_compute_*.txt")
    finally:
        analyzer._output.close()


def test_pre_processing_stdout_creates_no_file(tmp_path, monkeypatch) -> None:
    """--output-format stdout routes to the terminal and touches no file."""
    common.patch_console(monkeypatch, MODULE, "debug", "log", "warning")
    monkeypatch.setattr(OmniAnalyze_Base, "initalize_runs", lambda self: {})
    monkeypatch.chdir(tmp_path)

    analyzer = OmniAnalyze_Base(
        argparse.Namespace(
            output_format="stdout", output_name=None, **PRE_PROCESSING_ARGS
        ),
        {},
    )
    analyzer.pre_processing()

    assert analyzer._output is sys.stdout
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# initalize_runs --specs-correction handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("specs_correction", [None, ["num_xcd:4"]])
def test_initalize_runs_corrects_specs_only_when_asked(
    tmp_path, monkeypatch, specs_correction
) -> None:
    """Without --specs-correction the recorded sysinfo.csv is what analysis runs on."""
    sysinfo = {
        "ip_blocks": "SQ|LDS|TCC|roofline",
        "gpu_arch": "gfx950",
        "num_xcd": 8,
    }
    pd.DataFrame([sysinfo]).to_csv(tmp_path / "sysinfo.csv", index=False)
    corrected = pd.DataFrame([{**sysinfo, "num_xcd": "4"}])
    monkeypatch.setattr(f"{MODULE}.parser.correct_sys_info", lambda *_args: corrected)

    analyzer = OmniAnalyze_Base(
        argparse.Namespace(
            path=[[str(tmp_path)]],
            specs_correction=specs_correction,
            no_roof=True,
            normal_unit="per_kernel",
            list_stats=False,
            filter_metrics=None,
            gpu_kernel=None,
        ),
        {},
    )
    # Panel config generation reads the real arch YAML, which this test is not about.
    monkeypatch.setattr(analyzer, "generate_configs", lambda *_args: {})
    analyzer._arch_configs = {sysinfo["gpu_arch"]: SimpleNamespace(dfs={}, dfs_type={})}
    analyzer.set_soc({sysinfo["gpu_arch"]: SimpleNamespace(_mspec=object())})

    workload = analyzer.initalize_runs()[str(tmp_path)]

    expected_num_xcd = "4" if specs_correction else 8
    assert workload.sys_info["num_xcd"].item() == expected_num_xcd


# =============================================================================
# membw_analysis_collected tests
# =============================================================================


@pytest.mark.parametrize(
    "profiling_config,expected",
    [
        pytest.param({"membw_analysis": True}, True, id="collected"),
        pytest.param({"membw_analysis": False}, False, id="not_collected"),
        pytest.param({}, False, id="absent"),
    ],
)
def test_membw_analysis_collected(profiling_config, expected) -> None:
    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst._profiling_config = profiling_config
    assert inst.membw_analysis_collected() is expected


def test_membw_analysis_collected_without_config_attribute() -> None:
    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    assert inst.membw_analysis_collected() is False


def test_sanitize_errors_when_workload_profiled_with_kernel_multiplexing(
    tmp_path, monkeypatch
):
    workload = tmp_path / "app" / "MI300"
    workload.mkdir(parents=True)
    (workload / "profiling_config.yaml").write_text(
        "iteration_multiplexing: kernel\n", encoding="utf-8"
    )
    mock_error = common.patch_console(
        monkeypatch, MODULE, "error", error=Mock(side_effect=SystemExit(1))
    )["error"]
    with pytest.raises(SystemExit):
        OmniAnalyze_Base(
            argparse.Namespace(tui=False, path=[[str(workload)]]), {}
        ).sanitize()
    assert mock_error.call_args.args == (
        "analysis",
        'This workload was profiled with "--iteration-multiplexing kernel", '
        "which is no longer supported. "
        'Profile it again with "--iteration-multiplexing".',
    )


@pytest.mark.parametrize(
    ("listing", "operator", "expected"),
    [
        pytest.param(["torch"], None, None, id="torch-list"),
        pytest.param(["user"], None, None, id="user-list"),
        pytest.param([], ["triton:*matmul*"], None, id="listing-wins"),
        pytest.param([], ["invalid:*"], None, id="listing-ignores-invalid-prefix"),
        pytest.param(None, ["torch:*relu*"], {"torch": ["*relu*"]}, id="torch"),
        pytest.param(
            None,
            ["*relu*"],
            {"torch": ["*relu*"], "user": ["*relu*"]},
            id="unscoped",
        ),
        pytest.param(None, [], {"torch": ["**"], "user": ["**"]}, id="bare"),
    ],
)
def test_bind_operator_selectors_authorizes_trace_ml_api_selectors(
    tmp_path, listing, operator, expected
) -> None:
    """Bind selectors to recorded backends and preserve raw CLI tokens."""
    workload = write_traced_workload(tmp_path, "app", "torch")
    write_counter_results(workload)
    args = argparse.Namespace(
        tui=False,
        path=[[str(workload)]],
        list_operators=None if listing is None else list(listing),
        operator=None if operator is None else list(operator),
        output_name=None,
    )
    analyzer = OmniAnalyze_Base(args, {})

    analyzer.sanitize()
    assert analyzer._operator_patterns_by_path == {}
    analyzer.bind_operator_selectors(str(workload))

    assert args.operator == operator
    assert args.list_operators == listing
    expected_bindings = {} if expected is None else {str(workload.resolve()): expected}
    assert analyzer._operator_patterns_by_path == expected_bindings


@pytest.mark.parametrize(
    ("listing", "operator"),
    [
        pytest.param(["triton"], None, id="listing"),
        pytest.param(None, ["triton:*matmul*"], id="filter"),
    ],
)
def test_bind_operator_selectors_rejects_untraced_backend(
    tmp_path, monkeypatch, listing, operator
):
    workload = write_traced_workload(tmp_path, "app", "torch")
    write_counter_results(workload)
    errors = []
    monkeypatch.setattr(
        "ml_api_tracing.analysis.console_error",
        partial(record_error_and_exit, errors),
    )
    args = argparse.Namespace(
        tui=False,
        path=[[str(workload)]],
        list_operators=listing,
        operator=operator,
        output_name=None,
    )

    analyzer = OmniAnalyze_Base(args, {})
    analyzer.sanitize()
    with pytest.raises(SystemExit):
        analyzer.bind_operator_selectors(str(workload))

    assert "was not profiled" in errors[-1][1]
    assert "triton" in errors[-1][1]


def test_bind_operator_selectors_mixed_paths_bare_operator_accepts_either_order(
    tmp_path,
) -> None:
    """Bare --operator on sn_torch + tf_triton is legal in both -p orders."""
    torch_workload = write_traced_workload(tmp_path, "sn_torch", "torch")
    triton_workload = write_traced_workload(tmp_path, "tf_triton", "triton")
    for workload in (torch_workload, triton_workload):
        write_counter_results(workload)
    for paths in mixed_path_orders(torch_workload, triton_workload):
        args = argparse.Namespace(
            tui=False,
            path=paths,
            list_operators=None,
            operator=[],
            output_name=None,
        )
        analyzer = OmniAnalyze_Base(args, {})
        analyzer.sanitize()
        assert analyzer._operator_patterns_by_path == {}
        for path in paths:
            analyzer.bind_operator_selectors(path[0])
        assert args.operator == []
        torch_path = str(torch_workload.resolve())
        triton_path = str(triton_workload.resolve())
        assert analyzer._operator_patterns_by_path[torch_path] == {
            "torch": ["**"],
            "user": ["**"],
        }
        assert analyzer._operator_patterns_by_path[triton_path] == {
            "triton": ["**"],
            "user": ["**"],
        }


def test_bind_operator_selectors_mixed_paths_list_torch_rejects_either_order(
    tmp_path, monkeypatch
) -> None:
    """--list-operators torch on sn_torch + tf_triton fails in both -p orders."""
    torch_workload = write_traced_workload(tmp_path, "sn_torch", "torch")
    triton_workload = write_traced_workload(tmp_path, "tf_triton", "triton")
    for workload in (torch_workload, triton_workload):
        write_counter_results(workload)
    errors = []
    monkeypatch.setattr(
        "ml_api_tracing.analysis.console_error",
        partial(record_error_and_exit, errors),
    )
    for paths in mixed_path_orders(torch_workload, triton_workload):
        args = argparse.Namespace(
            tui=False,
            path=paths,
            list_operators=["torch"],
            operator=None,
            output_name=None,
        )
        analyzer = OmniAnalyze_Base(args, {})
        analyzer.sanitize()
        with pytest.raises(SystemExit):
            for path in paths:
                analyzer.bind_operator_selectors(path[0])
        assert "was not profiled" in errors[-1][1]
        assert "tf_triton" in errors[-1][1]


def test_sanitize_resets_path_bindings_and_keeps_raw_patterns_owned(
    tmp_path,
):
    torch_workload = write_traced_workload(tmp_path, "first", "torch")
    triton_workload = write_traced_workload(tmp_path, "second", "triton")
    for workload in (torch_workload, triton_workload):
        write_counter_results(workload)
    args = argparse.Namespace(
        tui=False,
        path=[[str(torch_workload)]],
        list_operators=None,
        operator=[],
        output_name=None,
    )
    analyzer = OmniAnalyze_Base(args, {})
    analyzer.sanitize()
    torch_path = str(torch_workload.resolve())
    analyzer.bind_operator_selectors(torch_path)
    analyzer._operator_patterns_by_path[torch_path]["torch"].append("copy-only")
    assert args.operator == []
    args.path = [[str(triton_workload)]]

    analyzer.sanitize()
    assert analyzer._operator_patterns_by_path == {}
    analyzer.bind_operator_selectors(str(triton_workload))

    assert analyzer._operator_patterns_by_path == {
        str(triton_workload.resolve()): {"triton": ["**"], "user": ["**"]}
    }
    assert args.operator == []
    args.list_operators = []
    args.operator = ["invalid-prefix:ignored"]
    analyzer.sanitize()
    assert analyzer._operator_patterns_by_path == {}
    analyzer.bind_operator_selectors(str(triton_workload))
    assert analyzer._operator_patterns_by_path == {}
    assert args.operator == ["invalid-prefix:ignored"]
