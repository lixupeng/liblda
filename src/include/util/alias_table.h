#pragma once

#include <cmath>
#include <vector>
#include "util/random.h"

namespace liblda {

class AliasTable {
 public:
  void Init(std::vector<double> &p) {
    n = static_cast<int>(p.size());
    table.resize(n);
    prob_sum = 0;

    for (int i = 0; i < n; ++i) {
      prob_sum += p[i];
    }
    prob_per_column = prob_sum / p.size();

    for (int i = 0; i < n; ++i) {
      table[i].less = table[i].greater = i;
      table[i].prob = p[i];
    }
    for (int i = 0, j = 0; i < n; ++i) {
      if (fabs(table[i].prob - prob_per_column) < 1e-10) {
        if (i > j) {
          Column tmp = table[i];
          table[i] = table[j];
          table[j] = tmp;
        }
        ++j;
        continue;
      }
      if (table[i].prob > prob_per_column == table[j].prob > prob_per_column)
        continue;
      if (table[i].prob > prob_per_column) {
        table[j].greater = table[i].greater;
        table[i].prob -= prob_per_column - table[j].prob;
      } else {
        double new_prob = table[j].prob + table[i].prob - prob_per_column;
        table[j].prob = table[i].prob;
        table[j].less = table[i].less;
        table[i].prob = new_prob;
        table[i].less = table[i].greater = table[j].greater;
      }
      --i, ++j;
    }
  }

  int Sample(Random &R) {
    double p = R.RandDouble(prob_sum);
    int n = std::min((int) (p / prob_per_column), n - 1);
    p -= n * prob_per_column;
    if (p < table[n].prob)
      return table[n].less;
    else
      return table[n].greater;
  }

 private:
  struct Column {
    double prob;
    int less;
    int greater;
  };

  std::vector<Column> table;
  double prob_sum, prob_per_column;
  int n;
};

}  // namespace liblda
