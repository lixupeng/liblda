#pragma once

#include <vector>

namespace liblda {

class FTree {
 private:
  std::vector<double> value;
  int n = 0;
  int topic_num = 0;

 public:
  double Get(int i) {
    return value[n + i];
  }

  double Sum() {
    return value[1];
  }

  void Set(int i, double val) {
    value[n + i] = val;
  }

  // build must be called after values set
  void Build() {
    for (int i = n - 1; i >= 1; --i) {
      value[i] = value[i + i] + value[i + i + 1];
    }
  }

  void Update(int i, double val) {
    i += n;
    value[i] = val;
    while (i > 1) {
      i >>= 1;
      value[i] = value[i + i] + value[i + i + 1];
    }
  }

  int Sample(double prob) {
    int i = 1;
    while (i < n) {
      if (prob < value[i + i]) {
        i = i + i;
      } else {
        prob -= value[i + i];
        i = i + i + 1;
      }
    }
    return std::min(i - n, topic_num - 1);
  }

  void Resize(int size) {
    n = 1;
    while (n < size) n <<= 1;
    value.resize(n * 2);
  }

  explicit FTree(int size) : topic_num(size) {
    Resize(size);
  }

  FTree() = default;
};

}
