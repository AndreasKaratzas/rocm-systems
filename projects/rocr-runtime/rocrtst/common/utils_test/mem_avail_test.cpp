/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Unit tests for common.h's MemAvailDelta(), MemAvailDisturbed() and
// MemAvailLeaked(). Pure integer functions, so these need no device.

#include <stddef.h>
#include <stdint.h>

#include "gtest/gtest.h"

#include "common/common.h"

using rocrtst::kMemAvailGranule;
using rocrtst::kMemAvailTolerance;
using rocrtst::MemAvailDelta;
using rocrtst::MemAvailDisturbed;
using rocrtst::MemAvailLeaked;

// The granule must be a non-zero power of two, and the tolerance must cover
// both readings yet stay small enough to leave a real neighbour detectable.
TEST(MemAvailTest, ConstantsAreUsableAsARoundingAllowance) {
  EXPECT_GT(kMemAvailGranule, 0u);
  EXPECT_EQ(kMemAvailGranule & (kMemAvailGranule - 1), 0u);

  EXPECT_GE(kMemAvailTolerance, kMemAvailGranule);
  EXPECT_LT(kMemAvailTolerance, 64ULL * 1024 * 1024);
}

// Two identical readings differ by nothing.
TEST(MemAvailTest, DeltaOfEqualReadingsIsZero) {
  EXPECT_EQ(MemAvailDelta(0, 0), 0u);
  EXPECT_EQ(MemAvailDelta(kMemAvailGranule, kMemAvailGranule), 0u);
}

// The readings are unsigned, so a plain `before - after` underflows to near
// 2^64 whenever availability rose rather than fell.
TEST(MemAvailTest, DeltaDoesNotUnderflowWhenAvailabilityRises) {
  const size_t before = 4096;
  const size_t after = 4096 + kMemAvailTolerance + 1;

  EXPECT_EQ(MemAvailDelta(before, after), kMemAvailTolerance + 1);
  EXPECT_LT(MemAvailDelta(before, after), before + after);
}

// The helper reports a magnitude, so argument order must not change it.
TEST(MemAvailTest, DeltaIsSymmetric) {
  const size_t lo = 1ULL << 20;
  const size_t hi = 1ULL << 40;

  EXPECT_EQ(MemAvailDelta(lo, hi), MemAvailDelta(hi, lo));
  EXPECT_EQ(MemAvailDelta(lo, hi), hi - lo);
}

// The arithmetic must hold at the top of the range, not only near zero.
TEST(MemAvailTest, DeltaHoldsAtTheExtremes) {
  const size_t max = SIZE_MAX;

  EXPECT_EQ(MemAvailDelta(max, max), 0u);
  EXPECT_EQ(MemAvailDelta(max, 0), max);
  EXPECT_EQ(MemAvailDelta(0, max), max);
}

// The comparison is strict: exactly the tolerance is still rounding, one byte
// more is a disturbance.
TEST(MemAvailTest, DisturbedIsFalseAtToleranceAndTrueJustAboveIt) {
  const size_t base = 64ULL * 1024 * 1024;

  EXPECT_FALSE(MemAvailDisturbed(base, base));
  EXPECT_FALSE(MemAvailDisturbed(base, base - kMemAvailTolerance));
  EXPECT_FALSE(MemAvailDisturbed(base, base + kMemAvailTolerance));

  EXPECT_TRUE(MemAvailDisturbed(base, base - kMemAvailTolerance - 1));
  EXPECT_TRUE(MemAvailDisturbed(base, base + kMemAvailTolerance + 1));
}

// A neighbour that frees disturbs the readings as much as one that allocates,
// so detection must not be one-sided.
TEST(MemAvailTest, DisturbedDetectsMovementInBothDirections) {
  const size_t base = 8ULL * 1024 * 1024 * 1024;
  const size_t neighbour = 1ULL * 1024 * 1024 * 1024;

  EXPECT_TRUE(MemAvailDisturbed(base, base - neighbour));
  EXPECT_TRUE(MemAvailDisturbed(base, base + neighbour));
}

// Sub-granule movement is rounding, never another process on the device.
TEST(MemAvailTest, SubGranuleMovementIsNotADisturbance) {
  const size_t base = 1ULL << 32;

  for (size_t drift = 0; drift <= kMemAvailGranule; drift += kMemAvailGranule / 4) {
    EXPECT_FALSE(MemAvailDisturbed(base, base - drift)) << "drift " << drift;
    EXPECT_FALSE(MemAvailDisturbed(base, base + drift)) << "drift " << drift;
  }
}

// A shortfall the size of one of the test's own allocations is that test
// failing to give memory back, not a neighbour, and must not be excused.
TEST(MemAvailTest, LeakedMatchesASingleOwnAllocation) {
  const size_t base = 68472012800ULL;
  const size_t alloc1 = 64ULL * 1024 * 1024;
  const size_t alloc2 = 512ULL * 1024 * 1024;

  EXPECT_TRUE(MemAvailLeaked(base, base - alloc1, {alloc1, alloc2}));
  EXPECT_TRUE(MemAvailLeaked(base, base - alloc2, {alloc1, alloc2}));
}

// Both allocations can leak together, so the running sum is a candidate too.
TEST(MemAvailTest, LeakedMatchesTheSumOfTheOwnAllocations) {
  const size_t base = 68472012800ULL;
  const size_t alloc1 = 64ULL * 1024 * 1024;
  const size_t alloc2 = 512ULL * 1024 * 1024;

  EXPECT_TRUE(MemAvailLeaked(base, base - (alloc1 + alloc2), {alloc1, alloc2}));
}

// The quantisation allowance applies to the match as well as to the reading.
TEST(MemAvailTest, LeakedToleratesOneGranuleEitherSideOfTheMatch) {
  const size_t base = 68472012800ULL;
  const size_t alloc = 64ULL * 1024 * 1024;

  EXPECT_TRUE(MemAvailLeaked(base, base - alloc - kMemAvailTolerance, {alloc}));
  EXPECT_TRUE(MemAvailLeaked(base, base - alloc + kMemAvailTolerance, {alloc}));
  EXPECT_FALSE(MemAvailLeaked(base, base - alloc - kMemAvailTolerance - 1, {alloc}));
}

// A neighbour's footprint bears no relation to this test's allocation sizes,
// so an unexplained shortfall stays a skip rather than becoming a failure.
TEST(MemAvailTest, LeakedIsFalseForAnUnrelatedShortfall) {
  const size_t base = 68472012800ULL;
  const size_t alloc = 64ULL * 1024 * 1024;
  const size_t neighbour = 270ULL * 1024 * 1024;

  EXPECT_FALSE(MemAvailLeaked(base, base - neighbour, {alloc}));
}

// Losing memory is the only way a leak can show, so a figure that rose is
// always foreign however well the magnitude happens to match.
TEST(MemAvailTest, LeakedIsFalseWhenAvailabilityRose) {
  const size_t base = 68472012800ULL;
  const size_t alloc = 64ULL * 1024 * 1024;

  EXPECT_FALSE(MemAvailLeaked(base, base + alloc, {alloc}));
}

// A test that allocated nothing can never account for a drift out of its own
// allocations, so the empty list must never produce a failure.
TEST(MemAvailTest, LeakedIsFalseWithNoOwnAllocations) {
  const size_t base = 68472012800ULL;

  EXPECT_FALSE(MemAvailLeaked(base, base - 64ULL * 1024 * 1024, {}));
}
