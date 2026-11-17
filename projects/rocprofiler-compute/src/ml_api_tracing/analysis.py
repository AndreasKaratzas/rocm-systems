# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Workload integration and diagnostics for ML API trace analysis."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from ml_api_tracing import call_tree
from ml_api_tracing.call_tree import (
    nest_marker_intervals,
    prune_cpu_only_call_trees,
    rollup_node_stats,
    validate_all_markers_nested,
)
from ml_api_tracing.errors import MlApiTraceError, UnaccountedKernelError
from ml_api_tracing.input import (
    load_joined_marker_counter_pass,
    load_ml_api_trace_pairs,
)
from ml_api_tracing.models import CallTreeForest
from ml_api_tracing.processing import (
    apply_parsed_function_columns,
    collapse_matching_markers_across_passes,
    group_marker_pass,
    unmatched_kernel_rows,
)
from ml_api_tracing.selection import (
    analyze_dispatch_ids_from_pmc_keys,
    assign_kernel_ids_from_top,
    intersect_dispatch_ids,
    intersect_dispatch_ids_by_kernel,
    select_operators,
)
from ml_api_tracing.selectors import (
    bind_operator_patterns,
    check_backends_traced,
    operator_backend_labels,
    traced_frameworks_from_config,
)
from utils import file_io, schema
from utils.logger import console_error, console_log, console_warning, demarcate


@demarcate
def process_ml_api_trace_output(
    workload: schema.Workload,
    workload_dir: str,
) -> None:
    """Populate a Workload with paired traces, parsed markers and call trees.

    Completed stages remain visible in ml_api_trace_pairs,
    unmatched_kernel_frames, ml_api_trace_df and ml_api_call_trees if a later
    stage fails. With no input pairs, only ml_api_trace_pairs is cleared.
    Integrity failures abort analyze; coverage issues are reported as warnings.
    """
    console_log(f"Looking for marker and counter csv files in {workload_dir}")
    try:
        pairs = load_ml_api_trace_pairs(Path(workload_dir))
        if not pairs:
            console_warning(
                "No marker files with corresponding counter files found. "
                "Ensure profiling was done with ML API tracing enabled "
                "(e.g., via '--trace-ml-api')."
            )
            workload.ml_api_trace_pairs = []
            return
        workload.ml_api_trace_pairs = pairs
        _build_ml_api_call_trees(workload)
    except MlApiTraceError as error:
        console_error("analysis", str(error))


def validate_operator_trace_options(
    workload_path: str,
    list_operators: Optional[List[str]],
    operator_patterns: Optional[List[str]],
    profiling_config: Dict[str, Any],
) -> Optional[Dict[str, List[str]]]:
    """Bind raw operator patterns to this workload's recorded frameworks.

    Listing takes precedence, including a bare list-all request. Errors name
    the workload path; neither the raw patterns nor profiling_config is changed.
    """
    if list_operators is None and operator_patterns is None:
        return None
    traced_frameworks = traced_frameworks_from_config(profiling_config)
    try:
        if list_operators is not None:
            check_backends_traced(list_operators, traced_frameworks)
            return None
        if operator_patterns is not None:
            return bind_operator_patterns(operator_patterns, traced_frameworks)
    except ValueError as error:
        console_error("ml api trace", f"{workload_path}: {error}")
    return None


def apply_operator_filter(
    workload: schema.Workload,
    patterns_by_backend: Optional[Dict[str, List[str]]],
    workload_path: str,
) -> List[Tuple]:
    """Replace workload.ml_api_call_trees with a copy of the matched operators.

    No match leaves the original forest in place and aborts analysis. A match
    without dispatches publishes the selected forest before reporting an error.
    Returned PMC keys are unique in their recorded order; kernel statistics
    keep their recorded launch counts and repeated keys.
    """
    if patterns_by_backend is None:
        return []
    labels = operator_backend_labels(list(patterns_by_backend))
    selected_forest, pmc_group_keys = select_operators(
        workload.ml_api_call_trees, patterns_by_backend
    )
    if not selected_forest:
        pattern_text = ", ".join(
            dict.fromkeys(
                pattern
                for patterns in patterns_by_backend.values()
                for pattern in patterns
            )
        )
        console_error(
            "ml api trace",
            f"{workload_path}: No {labels} operators matched the "
            f"pattern(s): {pattern_text}",
        )
    workload.ml_api_call_trees = selected_forest
    if not pmc_group_keys:
        console_error(
            "ml api trace",
            f"{workload_path}: No {labels} kernel dispatches on the matched operators",
        )
    return pmc_group_keys


def select_operator_dispatches(
    workload: schema.Workload,
    pmc_group_keys: List[Tuple],
    slot_to_id: Dict[Tuple, int],
    backends: List[str],
) -> None:
    """Publish selected dispatch IDs only after mapping and restriction succeed."""
    labels = operator_backend_labels(backends)
    try:
        operator_dispatch_ids = analyze_dispatch_ids_from_pmc_keys(
            pmc_group_keys, slot_to_id
        )
    except KeyError as error:
        console_error(
            "ml api trace",
            f"No {labels} analyze dispatch id for pmc group key {error.args[0]}",
        )
        return
    user_dispatch_filter = workload.filter_dispatch_ids
    requested_dispatch_ids, minimum_dispatch_id = _parse_operator_dispatch_filter(
        user_dispatch_filter
    )
    selected_ids = intersect_dispatch_ids(
        operator_dispatch_ids, requested_dispatch_ids, minimum_dispatch_id
    )
    if not selected_ids:
        console_error(
            "ml api trace",
            f"No {labels} dispatches matched the -d filter: {user_dispatch_filter}",
        )
    workload.filter_dispatch_ids = selected_ids
    console_log(
        "ml api trace",
        f"{labels} operator filter selected "
        f"{len(workload.filter_dispatch_ids)} dispatch(es) for metric analysis.",
    )


def apply_sliced_kernel_top(
    workload: schema.Workload,
    kernel_top_df: pd.DataFrame,
    args: argparse.Namespace,
    backends: List[str],
) -> None:
    """Number call-tree kernels from the filtered top-kernel table.

    If -k is present, restrict workload.filter_dispatch_ids to those kernels.
    Kernel numbering remains visible if the later filter validation fails.
    """
    labels = operator_backend_labels(backends)
    name_to_id = {
        str(kernel_name).strip(): idx
        for idx, kernel_name in enumerate(kernel_top_df["Kernel_Name"].tolist())
    }
    for roots in workload.ml_api_call_trees.values():
        for root in roots:
            assign_kernel_ids_from_top(root, name_to_id)
    if not workload.filter_kernel_ids:
        return
    file_io.validate_kernel_filter_ids(
        workload.filter_kernel_ids, len(kernel_top_df["Kernel_Name"])
    )
    workload.filter_dispatch_ids = intersect_dispatch_ids_by_kernel(
        workload.filter_dispatch_ids,
        workload.raw_pmc,
        kernel_top_df,
        workload.filter_kernel_ids,
    )
    if not workload.filter_dispatch_ids:
        console_error(
            "ml api trace",
            f"No {labels} operators matched the -k filter: {args.gpu_kernel}",
        )


def attach_unlocated_trees_by_launcher_thread(
    forest: CallTreeForest,
    errors: Optional[List[MlApiTraceError]] = None,
) -> None:
    """Attach worker roots with the adapter's warning policy.

    When errors is supplied, append issues without logging. Otherwise, log
    collected issues even if attachment raises.
    """
    issues = errors if errors is not None else []
    try:
        call_tree.attach_unlocated_trees_by_launcher_thread(forest, issues)
    finally:
        if errors is None:
            for issue in issues:
                console_warning("analysis", str(issue))


def _parse_operator_dispatch_filter(
    dispatch_filter: Optional[List[Union[str, int]]],
) -> Tuple[Optional[List[int]], Optional[int]]:
    """Interpret the existing explicit-ID or greater-than dispatch grammar."""
    if not dispatch_filter:
        return None, None
    first_filter = str(dispatch_filter[0]).strip()
    if first_filter.startswith(">"):
        return None, int(first_filter[1:].strip())
    return [int(value) for value in dispatch_filter], None


def _build_ml_api_call_trees(workload: schema.Workload) -> None:
    """Publish each completed join before advancing the Workload's next stage.

    Each pair's joined_df is assigned after its join and again after grouping.
    unmatched_kernel_frames is assigned once all joins finish. ml_api_trace_df
    is assigned after pass matching and parsing; ml_api_call_trees follows
    successful nesting. Later attachment or validation errors preserve this
    published state for diagnostics.
    """
    unmatched_frames: List[pd.DataFrame] = []
    for pair in workload.ml_api_trace_pairs:
        pair.joined_df = load_joined_marker_counter_pass(pair)
        unmatched_frames.append(unmatched_kernel_rows(pair.joined_df))
    workload.unmatched_kernel_frames = unmatched_frames
    nonempty_unmatched = [frame for frame in unmatched_frames if not frame.empty]
    if nonempty_unmatched:
        console_warning(
            "analysis",
            str(
                UnaccountedKernelError(pd.concat(nonempty_unmatched, ignore_index=True))
            ),
        )
    for pair in workload.ml_api_trace_pairs:
        pair.joined_df = group_marker_pass(pair.joined_df)
    skipped_keys: Set[Tuple[str, str]] = set()
    workload.ml_api_trace_df = apply_parsed_function_columns(
        collapse_matching_markers_across_passes([
            pair.joined_df for pair in workload.ml_api_trace_pairs
        ])
    )
    workload.ml_api_call_trees = nest_marker_intervals(
        workload.ml_api_trace_df, skipped_keys=skipped_keys
    )
    attach_unlocated_trees_by_launcher_thread(workload.ml_api_call_trees)
    validate_all_markers_nested(
        workload.ml_api_trace_df,
        workload.ml_api_call_trees,
        skipped_keys=skipped_keys,
    )
    prune_cpu_only_call_trees(workload.ml_api_call_trees)
    for roots in workload.ml_api_call_trees.values():
        for node in roots:
            rollup_node_stats(node)
    _record_missing_source_location_errors(workload.ml_api_call_trees)


def _record_missing_source_location_errors(
    forest: CallTreeForest,
    errors: Optional[List[MlApiTraceError]] = None,
) -> None:
    """Report missing source locations after rollup.

    When errors is supplied, append issues without logging. Otherwise, log
    collected issues even if detection raises.
    """
    issues = errors if errors is not None else []
    try:
        call_tree.record_missing_source_location_errors(forest, issues)
    finally:
        if errors is None:
            for issue in issues:
                console_warning("analysis", str(issue))
