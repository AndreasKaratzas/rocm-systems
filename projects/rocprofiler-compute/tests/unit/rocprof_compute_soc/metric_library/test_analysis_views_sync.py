# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Check that the analysis views keep the analysis configs' structure, arch by arch.

Temporary: this test is deleted with the analysis configs, once profile reads the views
too. Until then, each view is one analysis config panel:

- the same panels, tables, headers and table settings, in order (table ids are
  positions in the views, so they are not compared);
- the same rows in order, compared by the metric ids they show (the metric library sync
  test's row_metric_id() finds a panel row's ids), leaving out only rows that show no
  equation (and tables left empty by them);
- each panel's -b alias is the config template's.

Row texts are not compared: a view row shows its label or its metric's library name.
Values, units, peaks and descriptions come from the metric library and are not compared.
The check skips the panels the metric library sync test skips (the roofline panel of an
arch without a roofline benchmark), which gives no metric ids for their rows. The views
leave out the panels in LEFT_OUT_PANELS.
"""

import json
from typing import Any, Dict, List, Tuple

import common
import pytest
from unit.rocprof_compute_soc.metric_library.test_metric_library_sync import (
    read_rows,
    row_metric_id,
)

import config
from rocprof_compute_soc.metric_library import loader
from utils.analysis_views import (
    METRIC_TABLE,
    NO_VALUE,
    load_views,
    panel_aliases,
    row_metric_ids,
    views_to_panel_configs,
)
from utils.utils_common import (
    format_row_id,
    get_arch_panel_id_to_alias,
    load_panel_configs,
)

# Table settings that must match the panel's; rows are compared by metric id.
TABLE_KEYS = ("title", "header", "source", "columnwise")
# Row columns that show no equation.
NON_EQUATION_COLUMNS = ("unit", "peak", "pct_of_peak")
# Analysis config panels with no view: the SDK has no PC sampling on gfx908.
LEFT_OUT_PANELS = {"gfx908": (2100,)}


def structure_errors(
    views: Dict[int, Dict[str, Any]],
    library: loader.MetricLibrary,
    arch: str,
) -> List[str]:
    """Return how one arch's views differ from its analysis configs' structure."""
    row_ids = panel_row_metric_ids(arch, library)
    expected = load_panel_configs([str(config.analysis_configs_dir / arch)])
    for panel_id in LEFT_OUT_PANELS.get(arch, ()):
        del expected[panel_id]
    actual = views_to_panel_configs(views, library.metrics, {})
    for file_id in common.unshown_file_ids(arch):
        expected.pop(int(file_id), None)
        actual.pop(int(file_id), None)
    if list(expected) != list(actual):
        return [f"panels: analysis configs have {list(expected)}, views {list(actual)}"]
    errors = []
    for panel_id, panel in expected.items():
        errors.extend(
            f"panel {panel_id}: {error}"
            for error in _panel_errors(
                panel, actual[panel_id], views[panel_id], row_ids
            )
        )
    errors.extend(_alias_errors(views, arch))
    return errors


def panel_row_metric_ids(
    arch: str, library: loader.MetricLibrary
) -> Dict[str, List[str]]:
    """Return the metric ids each shown analysis config row of arch shows, by row
    id."""
    row_ids: Dict[str, List[str]] = {}
    for row in read_rows(arch):
        metric_id = row_metric_id(row, library, arch)
        row_ids.setdefault(row.legacy_id, [])
        if metric_id is not None:
            row_ids[row.legacy_id].append(metric_id)
    return row_ids


def _panel_errors(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    view: Dict[str, Any],
    row_ids: Dict[str, List[str]],
) -> List[str]:
    if expected["title"] != actual["title"]:
        return [f"title {actual['title']!r}, analysis configs {expected['title']!r}"]
    kept = [
        source
        for source in expected["data source"]
        if not _left_out_table(*_table(source))
    ]
    kinds = [list(source) for source in kept]
    if kinds != [list(source) for source in actual["data source"]]:
        return ["tables differ from the analysis configs' tables with rows"]
    errors = []
    for want, got, table in zip(kept, actual["data source"], view["tables"]):
        errors.extend(_table_errors(_table(want)[1], _table(got)[1], table, row_ids))
    return errors


def _table(source: Dict[str, Any]) -> Any:
    ((render, table),) = source.items()
    return render, table


def _left_out_table(render: str, table: Dict[str, Any]) -> bool:
    """Return whether a panel table is left out of the views: rows, none showing one."""
    rows = _table_rows(table) if render == METRIC_TABLE else {}
    return bool(rows) and not any(map(_shows_equation, rows.values()))


def _table_errors(
    want: Dict[str, Any],
    got: Dict[str, Any],
    view_table: Dict[str, Any],
    row_ids: Dict[str, List[str]],
) -> List[str]:
    where = f"table {want['id']}"
    errors = [
        f"{where}: {key} {got.get(key)!r}, analysis configs {want.get(key)!r}"
        for key in TABLE_KEYS
        if json.dumps(want.get(key)) != json.dumps(got.get(key))
    ]
    shown = [
        ids
        for name, row, ids in _rows_with_metric_ids(want, row_ids)
        if _shows_equation(row)
    ]
    got_ids = [row_metric_ids(row) for row in view_table.get("metrics", [])]
    if got_ids != shown:
        errors.append(f"{where}: rows {got_ids}, analysis configs {shown}")
    placeholder = (want.get("metric") or {}).get("placeholder_range")
    if placeholder != (got.get("metric") or {}).get("placeholder_range"):
        errors.append(f"{where}: placeholder_range differs")
    return errors


def _table_rows(table: Dict[str, Any]) -> Dict[str, Any]:
    rows = table.get("metric") or {}
    return {name: row for name, row in rows.items() if name != "placeholder_range"}


def _rows_with_metric_ids(
    table: Dict[str, Any], row_ids: Dict[str, List[str]]
) -> List[Tuple[str, Dict[str, Any], List[str]]]:
    """Return each row of a panel table with its metric ids; a row's id counts the
    placeholder_range key, as the metric library sync test's row ids do."""
    rows = table.get("metric") or {}
    return [
        (name, row, row_ids.get(format_row_id(table["id"], index), []))
        for index, (name, row) in enumerate(rows.items())
        if name != "placeholder_range"
    ]


def _shows_equation(row: Dict[str, Any]) -> bool:
    return any(
        str(value) != NO_VALUE
        for column, value in row.items()
        if column not in NON_EQUATION_COLUMNS
    )


def _alias_errors(views: Dict[int, Dict[str, Any]], arch: str) -> List[str]:
    template_aliases = get_arch_panel_id_to_alias(arch)
    return [
        f"panel {panel_number}: -b alias {alias!r}, config template has "
        f"{template_aliases.get(panel_number)!r}"
        for panel_number, alias in panel_aliases(views).items()
        if template_aliases.get(panel_number) != alias
    ]


def swap_rows(views: Dict[int, Dict[str, Any]]) -> None:
    table = views[1100]["tables"][1]
    table["metrics"][0], table["metrics"][1] = table["metrics"][1], table["metrics"][0]


def drop_row(views: Dict[int, Dict[str, Any]]) -> None:
    del views[1100]["tables"][1]["metrics"][2]


def drop_column(views: Dict[int, Dict[str, Any]]) -> None:
    del views[1100]["tables"][1]["header"]["max"]


def change_alias(views: Dict[int, Dict[str, Any]]) -> None:
    views[1100]["alias"] = "pipeline"


def arch_errors(arch: str, views: Dict[int, Dict[str, Any]]) -> List[str]:
    library = loader.MetricLibrary.load(loader.default_metric_library_dir(), arch)
    return structure_errors(views, library, arch)


@pytest.mark.parametrize("arch", list(loader.METRIC_LIBRARY_PARENT_ARCH))
def test_views_keep_the_panel_structure(arch):
    assert arch_errors(arch, load_views(arch)) == []


@pytest.mark.parametrize(
    "change, error",
    [
        (swap_rows, "panel 1100: table 1102: rows"),
        (drop_row, "panel 1100: table 1102: rows"),
        (drop_column, "panel 1100: table 1102: header"),
        (change_alias, "panel 11: -b alias 'pipeline', config template has 'cu_pipe'"),
    ],
)
def test_a_changed_structure_is_reported(change, error):
    """gfx942's views with one change no longer keep its panels' structure."""
    views = load_views("gfx942")
    change(views)
    errors = arch_errors("gfx942", views)
    assert any(e.startswith(error) for e in errors), errors
