/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

#include <map>
#include <sstream>
#include <string>

/**
 * @addtogroup hipInit hipInit
 * @{
 * @ingroup DriverTest
 */

/**
 * Test Description
 * ------------------------
 *  - In a fresh process, initialize HIP and query every device, then count the hardware queues the
 *    process holds, from KFD's per-process sysfs (/sys/class/kfd/kfd/proc/<pid>/queues).
 *    -# After init and device queries
 *      - Expected output: no queue on any GPU
 *    -# After a copy and a kernel on device 0
 *      - Expected output: queues on one GPU only
 *  - A queue created at init on every GPU costs each HIP process a doorbell, a runlist slot and
 *    VRAM on all GPUs, used or not (ROCM-34268: hipErrorInvalidValue under 64-way fork on MI350X
 *    CPX; ROCM-34585: out of VRAM on MI325X; ROCM-34588: oversubscribed hardware scheduler with
 *    16 ranks on MI250X).
 *  - KFD names the sysfs directory by the host PID; in a container the child finds its own by the
 *    VRAM of an allocation. Skipped where the directory cannot be found or lists no queues.
 * Test source
 * ------------------------
 *  - unit/device/hipInitQueues.cc
 * Test requirements
 * ------------------------
 *  - Linux, AMD
 */
HIP_TEST_CASE(Unit_hipInit_Positive_NoQueuesAtInit) {
  hip::SpawnProc proc("hipInitQueues", true, true);
  const int status = proc.run();
  const std::string output = proc.getOutput();
  INFO("hipInitQueues: " << output);
  REQUIRE(status == 0);

  std::map<std::string, int> result;
  std::istringstream tokens(output);
  std::string token;
  while (tokens >> token) {
    auto eq = token.find('=');
    if (eq != std::string::npos) result[token.substr(0, eq)] = std::stoi(token.substr(eq + 1));
  }
  REQUIRE(result.count("sysfs") == 1);
  if (result["sysfs"] == 0) {
    const char* reason = result["pidns"]
                             ? "own KFD sysfs directory not found from inside a PID namespace"
                             : "KFD per-process queue sysfs not available";
    HIP_SKIP_TEST(reason);
  }
  // Work on device 0 must show up as queues; otherwise sysfs does not list them and 0 proves
  // nothing.
  if (result["used_queues"] == 0) {
    HIP_SKIP_TEST("KFD sysfs lists no queues after a kernel launch");
  }

  REQUIRE(result["init_queues"] == 0);
  REQUIRE(result["init_gpus"] == 0);
  REQUIRE(result["used_gpus"] == 1);
}

/**
 * End doxygen group DriverTest.
 * @}
 */
