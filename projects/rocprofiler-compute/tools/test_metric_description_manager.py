# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for tools/config_management/metric_description_manager.py.

Run from the project directory:
    python3 -m pytest tools/test_metric_description_manager.py
"""

import shutil
from pathlib import Path

import yaml
from config_management.metric_description_manager import sync_arch

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs" / "data" / "metrics"

# A gfx115x section whose docs name differs from the name the generator gives it
VIEW_SECTION = "Memory chart - TCP Cache (GL0 Vector Cache)"
DOCS_SECTION = "Memory chart - TCP Cache"


def test_sync_keeps_hand_edited_rst_under_the_docs_section_name(tmp_path):
    shutil.copy(DOCS_DIR / "gfx115x_metrics.yaml", tmp_path)
    path = tmp_path / "gfx115x_metrics.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    metric = next(iter(data[DOCS_SECTION]))
    data[DOCS_SECTION][metric]["rst"] = "Edited by hand, see :ref:`desc-vmem`."
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    assert sync_arch("gfx115x", str(tmp_path))

    synced = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert VIEW_SECTION not in synced
    assert synced[DOCS_SECTION][metric]["rst"] == (
        "Edited by hand, see :ref:`desc-vmem`."
    )
    assert synced[DOCS_SECTION][metric]["plain"] == data[DOCS_SECTION][metric]["plain"]
