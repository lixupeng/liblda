#include <iostream>
#include <fstream>
#include "corpus/corpus.h"
#include "util/alias_table.h"
#include "model/lda.h"
#include "model/tot.h"
#include "model/ctm.h"
#include "model/slda.h"

namespace liblda {

Corpus &get_corpus(std::string path) {
  srand(19931214);
  static Corpus corpus;
  corpus.ReadFromFile(path);
  for (int d = 0; d < corpus.n_docs; ++d) {
    double time = (rand() % 65536) / 65536;
    corpus.doc_infos[d].emplace_back(time);
  }
  return corpus;
}

void test_lda(Corpus &corpus, int n_topics) {
  LdaModel lda(n_topics, 50.0 / n_topics, 0.01);
  lda.Train(corpus, 100);
}

void test_tot(Corpus &corpus, int n_topics) {
  TotModel tot(n_topics, 50.0 / n_topics, 0.01);
  tot.Train(corpus, 100);
}

void test_ctm(Corpus &corpus, int n_topics) {
  CtmModel ctm(n_topics, 0.0, 1.0, 0.01);
  ctm.Train(corpus, 100);
}

}

int main() {
  liblda::Corpus &corpus = liblda::get_corpus("/home/lixupeng/nips.train");
  liblda::test_ctm(corpus, 1000);
  return 0;
}
