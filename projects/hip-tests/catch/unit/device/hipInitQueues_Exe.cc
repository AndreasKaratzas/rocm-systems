/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Child of Unit_hipInit_Positive_NoQueuesAtInit. Reports, from KFD's per-process sysfs, the
// hardware queues this process holds after HIP init and after work on device 0, as "key=value"
// pairs:
//   sysfs        1 if this process's directory under /sys/class/kfd/kfd/proc was found
//   pidns        1 if the process is not in the initial PID namespace (a container)
//   devices      hipGetDeviceCount
//   init_queues  queues after hipInit, hipGetDeviceCount and hipGetDeviceProperties of every device
//   init_gpus    GPUs those queues are on
//   used_queues  queues after a copy and a kernel on device 0
//   used_gpus    GPUs those queues are on
// Returns 0 when it could report, 1 on an error (printed to stderr).

#include <hip/hip_runtime.h>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#define CHECK(call)                                                                                \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    if (err != hipSuccess) {                                                                       \
      std::cerr << #call << " failed: " << hipGetErrorString(err) << std::endl;                    \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

namespace {

const std::string kProcDir = "/sys/class/kfd/kfd/proc/";

struct Queues {
  int count = 0;
  std::set<std::string> gpus;
};

__global__ void addOne(int* data, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] += 1;
}

std::vector<std::string> listDir(const std::string& path) {
  std::vector<std::string> names;
  DIR* d = opendir(path.c_str());
  if (d == nullptr) return names;
  while (dirent* e = readdir(d)) {
    std::string name = e->d_name;
    if (name != "." && name != "..") names.push_back(name);
  }
  closedir(d);
  return names;
}

// Each queue of a process is a directory under <proc>/<pid>/queues holding the gpu_id of its GPU.
Queues readQueues(const std::string& pid) {
  Queues q;
  const std::string dir = kProcDir + pid + "/queues/";
  for (const auto& name : listDir(dir)) {
    std::ifstream f(dir + name + "/gpuid");
    std::string gpu;
    if (!(f >> gpu)) continue;
    ++q.count;
    q.gpus.insert(gpu);
  }
  return q;
}

// VRAM in use of every KFD process on every GPU, keyed by "<pid>/vram_<gpuid>".
std::map<std::string, long long> readVram() {
  std::map<std::string, long long> vram;
  for (const auto& pid : listDir(kProcDir)) {
    for (const auto& file : listDir(kProcDir + pid)) {
      if (file.rfind("vram_", 0) != 0) continue;
      std::ifstream f(kProcDir + pid + "/" + file);
      long long bytes = 0;
      if (f >> bytes) vram[pid + "/" + file] = bytes;
    }
  }
  return vram;
}

// KFD names a process's directory by its PID in the initial PID namespace, which getpid() does
// not return inside a container.
bool inInitialPidNamespace() {
  constexpr ino_t kInitPidNsIno = 0xEFFFFFFC;  // PROC_PID_INIT_INO
  struct stat st;
  return stat("/proc/self/ns/pid", &st) == 0 && st.st_ino == kInitPidNsIno;
}

// In a container, finds this process's directory as the only one whose VRAM grows by an
// allocation on device 0 and shrinks again when it is freed. hipMalloc creates no queue.
hipError_t findOwnDir(std::string* own) {
  constexpr size_t kBytes = 64 << 20;
  own->clear();
  for (int attempt = 0; attempt < 3 && own->empty(); ++attempt) {
    void* p = nullptr;
    auto before = readVram();
    hipError_t err = hipMalloc(&p, kBytes);
    if (err != hipSuccess) return err;
    auto allocated = readVram();
    err = hipFree(p);
    if (err != hipSuccess) return err;
    auto freed = readVram();
    std::set<std::string> dirs;
    for (const auto& [key, bytes] : allocated) {
      if (bytes - before[key] >= static_cast<long long>(kBytes) &&
          bytes - freed[key] >= static_cast<long long>(kBytes)) {
        dirs.insert(key.substr(0, key.find('/')));
      }
    }
    if (dirs.size() == 1) *own = *dirs.begin();
  }
  return hipSuccess;
}

}  // namespace

int main() {
  CHECK(hipInit(0));
  int devices = 0;
  CHECK(hipGetDeviceCount(&devices));
  for (int i = 0; i < devices; ++i) {
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, i));
  }

  // Queues of every KFD process now, before this process does anything on a device.
  std::map<std::string, Queues> init;
  for (const auto& pid : listDir(kProcDir)) init[pid] = readQueues(pid);

  const bool pidns = !inInitialPidNamespace();
  std::string own;
  if (!pidns) {
    own = std::to_string(getpid());
    if (init.count(own) == 0) own.clear();
  } else if (devices > 0) {
    CHECK(hipSetDevice(0));
    CHECK(findOwnDir(&own));
    if (init.count(own) == 0) own.clear();
  }
  if (own.empty() || devices == 0) {
    std::cout << "sysfs=0 pidns=" << pidns << " devices=" << devices << std::endl;
    return 0;
  }

  constexpr size_t n = 1024;
  std::vector<int> host(n, 1);
  int* dev = nullptr;
  CHECK(hipSetDevice(0));
  CHECK(hipMalloc(&dev, n * sizeof(int)));
  CHECK(hipMemcpy(dev, host.data(), n * sizeof(int), hipMemcpyHostToDevice));
  addOne<<<(n + 255) / 256, 256>>>(dev, n);
  CHECK(hipGetLastError());
  CHECK(hipMemcpy(host.data(), dev, n * sizeof(int), hipMemcpyDeviceToHost));
  CHECK(hipDeviceSynchronize());
  const Queues used = readQueues(own);
  CHECK(hipFree(dev));

  std::cout << "sysfs=1 pidns=" << pidns << " devices=" << devices
            << " init_queues=" << init[own].count << " init_gpus=" << init[own].gpus.size()
            << " used_queues=" << used.count << " used_gpus=" << used.gpus.size() << std::endl;
  return 0;
}
