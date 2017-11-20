/*
#pragma once

#include <atomic>
#include "util/sparse_counter.h"
#include "util/atomic_integer.h"
#include "util/random.h"
#include "corpus/corpus.h"

namespace liblda {

class SLdaModel {
 public:
  void Train(const Corpus &corpus, int max_iter = 100);
  void Inference(const Corpus &corpus);
  void LoadModel(const std::string &path);
  void SaveModel(const std::string &path);

  SLdaModel(int n_topics, double alpha, double beta,
            double mu, double nu2, double sigma2) :
      n_topics(n_topics), alpha(alpha), beta(beta),
      mu(mu), nu2(nu2), sigma2(sigma2) {
    eta.resize(n_topics);
  }

 private:
  int n_topics;
  double alpha, beta, mu, nu2, sigma2;
  std::vector<double> eta;

  std::unordered_map<std::string, int> word_to_int;
  std::vector<std::string> word_list;

  std::vector<SparseCounter> word_topic_dist;
  std::vector<AtomicInt> topic_dist;

  std::vector<double>

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
  double Loglikelihood(const Corpus &corpus);

  double CalDocTopic(int k, int Cdk, int Ck, double y_etaz, double eta_nk) {
    return exp(eta_nk * (2 * y_etaz - eta_nk) / (2 * sigma2)) *
        (Cdk + alpha) / (Ck + n_topics * beta);
  }
};

}
*/
