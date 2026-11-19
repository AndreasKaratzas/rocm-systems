# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""The HTML memory chart page and the values it shows."""

import json
import re
from pathlib import Path

import common
import pytest

from memory_chart import html_report
from memory_chart.html_report import chart_view, mem_chart_document
from memory_chart.loader import Layouts, PanelConfigs, layout_data
from memory_chart.render import edge_segments
from memory_chart.units import numeric, panel_units

MEMORY_CHART_ARCHS = common.memory_chart_archs()
ASSETS = Path(html_report.__file__).parent / "assets"


def embedded_model(document: str) -> dict:
    """The JSON model of a memory chart page."""
    match = re.search(
        r'<script id="mem-chart-model" type="application/json">(.*?)</script>',
        document,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


@pytest.mark.parametrize("arch", MEMORY_CHART_ARCHS)
def test_page_shows_what_the_terminal_chart_shows(arch):
    """Every value and stall on the page reads as in the terminal chart."""
    values = common.memory_chart_values(arch)
    layout = Layouts.for_arch(arch)
    level = next((b.stall_level for b in layout.blocks() if b.stall_level), None)
    membw = common.membw_result(common.bottleneck("Busy", level)) if level else None
    terminal = common.memory_chart_text(arch, values, membw)

    view = chart_view(arch, values, panel_units(PanelConfigs.for_arch(arch)), membw)

    assert set(view["values"]) == layout.metrics()
    for metric_id, shown in view["values"].items():
        assert shown["text"] in terminal
        assert shown["number"] == numeric(values.get(metric_id))
    stalls = [stall for rows in view["stalls"].values() for stall in rows]
    assert len(stalls) == (1 if level else 0)
    for stall in stalls:
        assert f"[!] {stall['label']} {stall['text']}" in terminal


def test_page_shows_guidance_or_the_status_line_as_the_terminal_does():
    """Active stalls with guidance show it; otherwise the page shows the status."""
    values = common.memory_chart_values("gfx950")
    units = panel_units(PanelConfigs.for_arch("gfx950"))
    guided = common.membw_result(
        common.bottleneck("Busy", "GL2"), guidance_blocks=("[GL2] Busy\n  Impact: x",)
    )
    idle = common.membw_result(common.bottleneck("Busy", "GL2", state="inactive"))

    with_guidance = chart_view("gfx950", values, units, guided)
    without_stalls = chart_view("gfx950", values, units, idle)
    without_membw = chart_view("gfx950", values, units, None)

    assert (with_guidance["status"], with_guidance["guidance"]) == (
        None,
        ["[GL2] Busy\n  Impact: x"],
    )
    assert without_stalls["status"].startswith("Memory Bandwidth Analysis: No ")
    assert without_stalls["guidance"] == []
    assert (without_membw["status"], without_membw["guidance"]) == (None, [])


def test_each_stall_links_to_its_guidance_block():
    """A stall points at the block rendered from its guidance id, if any."""
    membw = common.membw_result(
        common.bottleneck("Write stall", "GL2", guidance_id="write"),
        common.bottleneck("Hit rate", "GL2", guidance_id="no_template"),
        guidance_blocks=("[GL2] Write\n  Impact: x",),
        guidance_block_ids=("write",),
    )

    view = chart_view(
        "gfx950",
        common.memory_chart_values("gfx950"),
        panel_units(PanelConfigs.for_arch("gfx950")),
        membw,
    )

    assert [(s["label"], s["guidance"]) for s in view["stalls"]["l2"]] == [
        ("Write stall", 0),
        ("Hit rate", None),
    ]


def test_document_embeds_the_layout_file_and_the_view():
    """The page draws the shipped layout file with the view it is given."""
    view = chart_view(
        "gfx950",
        common.memory_chart_values("gfx950"),
        panel_units(PanelConfigs.for_arch("gfx950")),
        None,
    )

    model = embedded_model(
        mem_chart_document(
            "gfx950", common.MEMORY_CHART_TITLE, PanelConfigs.for_arch("gfx950"), view
        )
    )

    layout = Layouts.for_arch("gfx950")
    blocks = {block.id: block for block in layout.blocks()}
    grouped = [i for edge in model["edges"] for group in edge for i in group["arrows"]]
    assert model["heading"] == common.MEMORY_CHART_TITLE
    # Each value names its metric and describes it, as the panel config does
    assert set(model["metrics"]) == layout.metrics()
    assert model["metrics"]["3.1.1"]["name"] == "VGPRs"
    assert all(metric["description"] for metric in model["metrics"].values())
    assert model["layout"] == layout_data("gfx950")
    assert model["views"]["aggregate"] == json.loads(json.dumps(view))
    # The page groups the arrows between columns as the terminal does
    assert [group["anchor"] for group in model["edges"][0]] == [
        anchor for anchor in edge_segments(layout)[0]
    ]
    assert sorted(grouped) == [
        i for i, arrow in enumerate(layout.arrows) if blocks[arrow.target].row == 0
    ]


def test_page_script_finds_every_element_it_uses():
    """Each id the script looks up is in the page body."""
    script = (ASSETS / "mem_chart.js").read_text(encoding="utf-8")
    body = (ASSETS / "mem_chart.html").read_text(encoding="utf-8")

    used = set(re.findall(r'getElementById\("([^"]+)"\)', script))

    assert used
    assert used <= set(re.findall(r'id="([^"]+)"', body))
