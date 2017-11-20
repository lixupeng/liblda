#pragma once

#include <atomic>
#include "util/sparse_counter.h"
#include "util/atomic_integer.h"
#include "util/random.h"
#include "corpus/corpus.h"

namespace liblda {

class LdaModel {
 public:
  void Train(const Corpus &corpus, int max_iter = 100);
  void Inference(const Corpus &corpus);
  void LoadModel(const std::string &path);
  void SaveModel(const std::string &path);

  LdaModel(int n_topics, double alpha, double beta) :
      n_topics(n_topics), alpha(alpha), beta(beta) {}

 private:
  int n_topics;
  double alpha, beta;
  std::unordered_map<std::string, int> word_to_int;
  std::vector<std::string> word_list;

  std::vector<std::vector<int>> word_topic_dist;
  std::vector<AtomicInt> topic_dist;

  std::vector<SparseCounter> doc_topic_dist;

  struct Token {
    int topic;
    std::vector<int> mh_step;
  };
  std::vector<Token> topics;

  const static int LONG_DOC_LENGTH = 1;
  const static int MH_STEP = 2;
  std::vector<bool> is_long_doc;

  Random rand[32];

  void PrepareForTrain(const Corpus &corpus);
  void FTreeIteration(const Corpus &corpus);
  void VisitByDoc(const Corpus &corpus);
  void VisitByWord(const Corpus &corpus);
  double Loglikelihood(const Corpus &corpus);
  double CalWordTopic(int word_count, int topic, int n_words) {
    return (word_count + beta) / (topic_dist[topic] + n_words * beta);
  }
};

}
