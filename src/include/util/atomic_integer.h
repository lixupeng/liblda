#pragma once

#include <atomic>

namespace liblda {

struct AtomicInt : std::atomic_int {
  AtomicInt() : std::atomic_int(0) {}
  AtomicInt(AtomicInt &&b) {
    this->store(b);
  }
};

}  // namespace liblda
