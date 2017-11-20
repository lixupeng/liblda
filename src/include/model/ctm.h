#pragma once

#include <atomic>
#include "util/sparse_counter.h"
#include "util/atomic_integer.h"
#include "util/random.h"
#include "util/numeric/eigenmvn.h"
#include "corpus/corpus.h"

namespace liblda {

class CtmModel {
 public:
  void Train(const Corpus &corpus, int max_iter = 100);
  void Inference(const Corpus &corpus);
  void LoadModel(const std::string &path);
  void SaveModel(const std::string &path);

  CtmModel(int n_topics, double mu, double sigma, double beta) :
      n_topics(n_topics), mu_(mu), sigma_(sigma), beta(beta) {}

 private:
  int n_topics;
  double mu_, sigma_, beta;
  Eigen::VectorXd mu;
  Eigen::MatrixXd sigma, inv_sigma;

  std::unordered_map<std::string, int> word_to_int;
  std::vector<std::string> word_list;

  std::vector<SparseCounter> word_topic_dist;
  std::vector<AtomicInt> topic_dist;

  Eigen::MatrixXd eta, exp_eta;

  struct Token {
    int topic;
    std::vector<int> mh_step;
  };
  std::vector<Token> topics;

  const static int LARGE_WORD_SIZE = 100;
  const static int MH_STEP = 2;
  const static int SGLD_ITER_NUM = 10;
  std::vector<bool> is_large_word;

  Random rand[32];

  void PrepareForTrain(const Corpus &corpus);
  void FTreeIteration(const Corpus &corpus);
  void VisitByDoc(const Corpus &corpus);
  void VisitByWord(const Corpus &corpus);
  void SampleOthers(const Corpus &corpus);
  double Loglikelihood(const Corpus &corpus);

  double CalDocTopic(int k, int Ck, double exp_eta, int n_words) {
    return exp_eta / (Ck + n_words * beta);
  }
};

}
