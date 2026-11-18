# Tests for the rocrtst common/ helpers that need no GPU; anything needing an
# agent, pool or queue belongs in rocrtst64 instead.

# utils_timer_gtest.cpp and utils_timer_test.cpp are not built: they do not
# compile, and the gtest case wrapping them asserts nothing.
set(rocrtstUtilsTestSrcs
  mem_avail_test.cpp
  poll_backoff_test.cpp
  utils_cpp11_gtest.cpp)

add_executable(${ROCRTST_UTIL_TEST_NAME} ${rocrtstUtilsTestSrcs})

get_filename_component(UTILS_TEST_HSA_RUNTIME_ROOT
  "${ROCRTST_ROOT}/../runtime/hsa-runtime" ABSOLUTE)

target_include_directories(${ROCRTST_UTIL_TEST_NAME} PRIVATE
  # "common/common.h".
  ${ROCRTST_ROOT}
  ${ROCRTST_ROOT}/gtest/include
  # "core/util/poll_backoff.h", the header under test in poll_backoff_test.cpp.
  ${UTILS_TEST_HSA_RUNTIME_ROOT}
  # "hsa/hsa.h", pulled in by common/common.h. Headers only: no runtime entry
  # point is called, so the binary does not link hsa-runtime64.
  $<TARGET_PROPERTY:hsa-runtime64::hsa-runtime64,INTERFACE_INCLUDE_DIRECTORIES>)

# ${GOOGLE_TEST_FRWK_NAME} also compiles src/gtest_main.cpp, so these sources
# supply no main() of their own.
target_link_libraries(${ROCRTST_UTIL_TEST_NAME} ${GOOGLE_TEST_FRWK_NAME} pthread)

# Installed from here, not the parent: install(TARGETS) could not name a target
# from another directory until CMake 3.13, and this project declares 3.12.
install(TARGETS ${ROCRTST_UTIL_TEST_NAME}
        RUNTIME DESTINATION bin)
