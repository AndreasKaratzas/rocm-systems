#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: set command."""

from copy import deepcopy

from common.common import amdsmi

from cli.base import TestCliBase


def _strip_prefix(value, prefix):
    # Backport of str.removeprefix() (added in Python 3.9) because the test suite
    # still supports Python 3.8. Strips a leading enum-name prefix (e.g.
    # "AMDSMI_DEV_PERF_LEVEL_") when present, otherwise returns value unchanged.
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


class TestSet(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi set"
        self.common.print(msg)

        restore_cmds = self._restore_commands()
        cmds = self.CreateCmds(
            "set", "Set Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.addCleanup(self.RunCmds, restore_cmds)
        self.RunCmds(cmds)

        return

    def _section(self, data: dict, key: str, *, index: int, default: object) -> object:
        is_apu = self.clk_freq[index]["is_apu"]
        value = data.get(key, default) if is_apu else data[key]
        if is_apu and value == "N/A":
            value = default
        if not isinstance(value, type(default)):
            raise TypeError(f"GPU {index}: invalid {key} section")
        return {**default, **value} if is_apu and isinstance(default, dict) else value

    def _restore_commands(self) -> list:
        """Validate the recorded data and prepare cleanup before the sweep writes."""
        metric_data = deepcopy(self.metric_data)
        static_data = deepcopy(self.static_data)
        for index in range(len(self.common.processors)):
            metric = metric_data["gpu_data"][index]
            # The APU default metric dump drops the fan section, N/A included.
            if self.clk_freq[index]["is_apu"] and "fan" in metric:
                self.fail(f"GPU {index}: APU metric dump has a fan section: {metric['fan']}")
            for key, default in (
                ("fan", {"max": "N/A", "speed": "N/A"}),
                ("clock", {}),
                ("perf_level", "N/A"),
            ):
                metric[key] = self._section(metric, key, index=index, default=default)
            static = static_data["gpu_data"][index]
            static["clock"] = self._section(static, "clock", index=index, default={"sys": "N/A"})
        cmds = []
        for index, gpu in enumerate(self.common.processors):
            # reset --fans (works for both legacy hwmon and gpu_od interfaces)
            # The driver hides pwm1_max only along with every other fan attribute,
            # pwm1_enable included, so a GPU without max cannot reset its fan. Speed is
            # not used: a runtime-suspended dGPU reports it as "N/A".
            fan_max = metric_data["gpu_data"][index]["fan"]["max"]
            if self.clk_freq[index]["is_apu"] or fan_max == "N/A":
                fan_reset = [amdsmi.AmdSmiStatus.NOT_SUPPORTED]
            else:
                # max fan speed is the hwmon pwm1_max ceiling on every interface
                self.assertGreater(fan_max, 0, f"GPU {index}: max fan speed must be > 0")
                self.assertLessEqual(fan_max, 255, f"GPU {index}: max fan speed must be <= 255")
                # A present fan is not necessarily controllable.
                fan_reset = [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED]
            cmds.append((f"amd-smi reset --fans --gpu {index}", fan_reset))

            # set --profile defaults
            # Read from the static baseline: the library is shut down once setup
            # ends, so querying it here fails and the restore was skipped.
            profile = static_data["gpu_data"][index].get("profile")
            if isinstance(profile, dict) and profile["current"] in profile["available_profiles"]:
                cmds.append(
                    (f"amd-smi set --profile {profile['current']} --gpu {index}", self.PASS)
                )

            # set --perf-determinism defaults
            clock_sys = static_data["gpu_data"][index]["clock"]["sys"]
            if clock_sys != "N/A" and clock_sys["frequency_levels"]:
                num = len(clock_sys["frequency_levels"])
                level = f"Level {num - 1}"
                clock_freq = int(clock_sys["frequency_levels"][level]["value"])
                # Readable clock levels do not imply the determinism feature exists.
                cmds.append(
                    (
                        f"amd-smi set --perf-determinism {clock_freq} --gpu {index}",
                        [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                    )
                )

            # set --memory-partition defaults
            # Safe to write back: the mode only takes effect after a driver reload.
            memory_partition = self.partition_data["current_partition"][index]["memory"]
            if memory_partition != "N/A":
                cmds.append(
                    (f"amd-smi set --memory-partition {memory_partition} --gpu {index}", self.PASS)
                )

            # set --compute-partition defaults
            # Must come after memory partition sets, since the memory partition set can override
            # this setting (their defaults may differ).
            accelerator_type = self.partition_data["current_partition"][index]["accelerator_type"]
            if accelerator_type != "N/A":
                cmds.append(
                    (f"amd-smi set --compute-partition {accelerator_type} --gpu {index}", self.PASS)
                )

            # set --compute-partition-mem-alloc-mode defaults
            try:
                mem_alloc_mode = static_data["gpu_data"][index]["partition"][
                    "compute_partition_mem_alloc_mode"
                ]
            except (KeyError, TypeError):
                mem_alloc_mode = "N/A"
            if mem_alloc_mode not in ("N/A", "INVALID"):
                cmds.append(
                    (
                        f"amd-smi set --compute-partition-mem-alloc-mode {mem_alloc_mode} --gpu {index}",
                        self.PASS,
                    )
                )

            # reset --power-cap
            # Writes each supported sensor's default_power_cap. Replaying the recorded
            # value did not restore it: min and max were written after it, so the cap
            # was left at max. The current cap is not read, so a cap of 0 still resets.
            power_limits = static_data["gpu_data"][index]["limit"]
            reports_cap = any(
                isinstance(power_limits[power_type], dict)
                and power_limits[power_type]["max_power_limit"] != "N/A"
                for power_type in self.power_types
            )
            cmds.append(
                (
                    f"amd-smi reset --power-cap --gpu {index}",
                    self.PASS if reports_cap else [amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                )
            )

            # set --soc-pstate defaults
            soc_pstate = static_data["gpu_data"][index]["soc_pstate"]
            if soc_pstate != "N/A":
                current = int(soc_pstate["current_id"])
                cmds.append((f"amd-smi set --soc-pstate {current} --gpu {index}", self.PASS))

            # set --xgmi-plpd defaults
            xgmi_plpd = static_data["gpu_data"][index]["xgmi_plpd"]
            if xgmi_plpd != "N/A":
                current = int(xgmi_plpd["current_id"])
                cmds.append((f"amd-smi set --xgmi-plpd {current} --gpu {index}", self.PASS))

            # set --ptl-status defaults
            ptl_state = static_data["gpu_data"][index]["limit"]["ptl_state"]
            if ptl_state != "N/A":
                if ptl_state == "Disabled":
                    ptl_state_value = 0
                else:
                    ptl_state_value = 1
                cmds.append(
                    (f"amd-smi set --ptl-status {ptl_state_value} --gpu {index}", self.PASS)
                )

            # set --ptl-format defaults
            ptl_format = static_data["gpu_data"][index]["limit"]["ptl_format"]
            if ptl_format != "N/A":
                # TODO: get the right ptl-format
                cmds.append((f"amd-smi set --ptl-format {ptl_format} --gpu {index}", self.PASS))

            # set --clk-limit defaults
            clock = metric_data["gpu_data"][index]["clock"]
            for clk_type in self.clk_limits:
                for limit_type in self.limit_types:
                    value = self._clk_limit_value(clock, clk_type, limit_type)
                    if value is not None:
                        # A readable min/max clock does not imply the device
                        # supports setting a limit on it.
                        cmds.append(
                            (
                                f"amd-smi set --clk-limit {clk_type} {limit_type} {value} --gpu {index}",
                                [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                            )
                        )

            # reset --clocks
            # A clk-level write is a bitmask, so replaying the recorded level would pin DPM
            # to that one level; only AUTO hands every level back. Overdrive is reset too,
            # and reports NOT_SUPPORTED on parts that lack it even though the rest worked.
            cmds.append(
                (
                    f"amd-smi reset --clocks --gpu {index}",
                    [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                )
            )

            # set --perf-level defaults
            # Last of the clock writes: the reset above forces AUTO, and --clk-limit and
            # --perf-determinism each leave the device on a level of their own.
            perf_level = metric_data["gpu_data"][index]["perf_level"]
            if perf_level not in ("N/A", "AMDSMI_DEV_PERF_LEVEL_UNKNOWN"):
                perf_level = _strip_prefix(perf_level, "AMDSMI_DEV_PERF_LEVEL_")
                cmds.append((f"amd-smi set --perf-level {perf_level} --gpu {index}", self.PASS))
            # set --process-isolation defaults
            process_isolation = static_data["gpu_data"][index]["process_isolation"]
            # Put back whatever value the sweep left; "N/A" means unreadable, so
            # there is nothing to restore.
            if process_isolation != "N/A":
                original = 0 if process_isolation == "Disabled" else 1
                cmds.append(
                    (
                        f"amd-smi set --process-isolation {original} --gpu {index}",
                        [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                    )
                )

        return cmds
