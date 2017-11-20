#pragma once

#include <atomic>
#include "util/sparse_counter.h"
#include "util/atomic_integer.h"
#include "util/random.h"
#include "corpus/corpus.h"

namespace liblda {

class TotModel {
 public:
  void Train(const Corpus &corpus, int max_iter = 100);
  void Inference(const Corpus &corpus);
  void LoadModel(const std::string &path);
  void SaveModel(const std::string &path);

  TotModel(int n_topics, double alpha, double beta) :
      n_topics(n_topics), alpha(alpha), beta(beta) {}

 private:
  int n_topics;
  double alpha, beta;

  std::unordered_map<std::string, int> word_to_int;
  std::vector<std::string> word_list;

  std::vector<SparseCounter> word_topic_dist;
  std::vector<AtomicInt> topic_dist;

  std::vector<double> phi_a, phi_b;
  std::vector<double> lgamma_phi;

  std::vector<double> doc_time, log_doc_time, log_1_doc_time;

  struct Token {
    int topic;
    std::vector<int> mh_step;
  };
  std::vector<Token> topics;

  const static int LARGE_WORD_SIZE = 100;
  const static int MH_STEP = 2;
  std::vector<bool> is_large_word;

  Random rand[32];

  void PrepareForTrain(const Corpus &corpus);
  void FTreeIteration(const Corpus &corpus);
  void VisitByDoc(const Corpus &corpus);
  void VisitByWord(const Corpus &corpus);
  void EstimatePhi(const Corpus &corpus);
  double Loglikelihood(const Corpus &corpus);

  double GetDocTime(const Corpus &corpus, int d) {
    return corpus.doc_infos[d][0].double_value;
  }

  double CalDocTopic(int k, int d, int Cdk, int Ck, int n_words) {
    return exp(CalLogBetaTime(d, k)) * (Cdk + alpha) / (Ck + n_words * beta);
  }

  double CalLogBetaTime(int doc, int topic) {
    double P = (phi_a[topic] - 1) * log_doc_time[doc] +
        (phi_b[topic] - 1) * log_1_doc_time[doc];
    return P - lgamma_phi[topic];
  }
};

}
