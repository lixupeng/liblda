/*
#include "model/slda.h"
#include "util/alias_table.h"
#include "util/ftree.h"
#include <omp.h>
#include <sys/time.h>

namespace liblda {

void SLdaModel::Train(const Corpus &corpus, int max_iter) {
  PrepareForTrain(corpus);
  struct timeval time_start, time_end;
  unsigned sample_time, total_time = 0;
  for (int i = 0; i < max_iter; ++i) {
    gettimeofday(&time_start, NULL);
    VisitByDoc(corpus);
    VisitByWord(corpus);
    FTreeIteration(corpus);
    gettimeofday(&time_end, NULL);
    sample_time = (unsigned int) ((time_end.tv_sec - time_start.tv_sec) * 1000000 + time_end.tv_usec - time_start.tv_usec);
    total_time += sample_time;
    std::cout << "iter " << i << " " << sample_time << " " << total_time << std::endl;
    std::cout << "llh: " << Loglikelihood(corpus) << std::endl;
  }
}

void SLdaModel::PrepareForTrain(const Corpus &corpus) {
  topics.clear();
  topics.resize(corpus.n_tokens);
  topic_dist.clear();
  topic_dist.resize(n_topics);
  is_large_word.resize(corpus.n_words);

  AliasTable alias;
  std::vector<double> dist(n_topics);

  for (int w = 0; w < corpus.n_words; ++w) {
    std::string word = corpus.word_list[w];
    is_large_word[w] = (corpus.GetWordSize(w) > LARGE_WORD_SIZE);

    bool new_word = false;
    if (word_to_int.count(word) == 0) {
      word_to_int[word] = static_cast<int>(word_list.size());
      word_list.emplace_back(word);
      word_topic_dist.emplace_back(n_topics);
      new_word = true;
    }
    else {
      SparseCounter &word_dist = word_topic_dist[word_to_int[word]];
      for (int i = 0; i < n_topics; ++i) {
        dist[i] = word_dist.Count(i);
      }
      alias.Init(dist);
    }

    int start = corpus.word_offset[w];
    int end = corpus.word_offset[w + 1];
    for (int i = start; i < end; ++i) {
      int idx = corpus.word_to_doc[i];
      int topic = 0;
      if (new_word) {
        topic = topics[idx].topic = rand[0].RandInt(n_topics);
        if (is_large_word[w]) {
          for (int j = 0; j < MH_STEP; ++j) {
            topics[idx].mh_step.push_back(rand[0].RandInt(n_topics));
          }
        }
      }
      else {
        topic = topics[idx].topic = alias.Sample(rand[0]);
        if (is_large_word[w]) {
          for (int j = 0; j < MH_STEP; ++j) {
            topics[idx].mh_step.push_back(alias.Sample(rand[0]));
          }
        }
      }
      topic_dist[topic]++;
    }
  }
  for (int w = 0; w < corpus.n_words; ++w) {
    std::string word = corpus.word_list[w];
    int id = word_to_int[word];
    if (id != w) {
      std::swap(word_list[id], word_list[w]);
      word_topic_dist[id].Swap(word_topic_dist[w]);
      word_to_int[word_list[id]] = id;
      word_to_int[word_list[w]] = w;
    }
  }
}

void SLdaModel::FTreeIteration(const Corpus &corpus) {
  FTree tree(n_topics);
  std::vector<double> psum, eta_n;
  std::vector<int> doc_dist, doc_dist_o;

#pragma omp parallel for schedule(dynamic) private(psum) private(tree) private(doc_dist) private(doc_dist_o) private(eta_n)
  for (int doc = 0; doc < corpus.n_docs; ++doc) {
    int thread = omp_get_thread_num();
    int N = corpus.GetDocSize(doc);

    eta_n.resize(n_topics);
    for (int k = 0; k < n_topics; ++k) {
      eta_n[k] = eta[k] / N;
    }

    doc_dist.clear();
    doc_dist.resize(n_topics);
    for (int i = corpus.doc_offset[doc]; i < corpus.doc_offset[doc + 1]; ++i) {
      doc_dist[topics[i].topic]++;
    }
    doc_dist_o = doc_dist;

    double y = corpus.doc_infos[doc][0].double_value;
    double eta_z = 0;
    for (int k = 0; k < n_topics; ++k) {
      eta_z += eta_n[k] * doc_dist[k];
    }
    double eta_z_o = eta_z;

    for (int k = 0; k < n_topics; ++k) {
      tree.Set(k, CalDocTopic(k, doc_dist[k], topic_dist[k], y - eta_z, eta_n[k]));
    }
    tree.Build();
    for (int i = corpus.doc_offset[doc]; i < corpus.doc_offset[doc]; ++i) {
      int word = corpus.docs[i];
      if (is_large_word[word]) continue;
      int topic = topics[i].topic;
      SparseCounter &word_dist = word_topic_dist[doc];
      word_dist.Lock();

      doc_dist[topic]--;
      word_dist.Dec(topic);
      topic_dist[topic]--;
      eta_z -= eta_n[topic];
      tree.Update(topic, CalDocTopic(topic, doc_dist[topic], topic_dist[topic],
                                     y - eta_z_o, eta_n[topic]));

      double prob_left = tree.Sum() * beta;
      double prob_all = prob_left;
      psum.resize(n_topics);
      const std::vector<CountItem> &items = word_dist.GetItem();
      for (int t = 0, s = (int)items.size(); t < s; ++t) {
        double p = items[t].count * tree.Get(items[t].item);
        prob_all += p;
        psum[t] = p;
        if (t > 0) psum[t] += psum[t - 1];
      }

      double prob = rand[thread].RandDouble(prob_all);
      int new_topic;
      if (prob < prob_left) {
        new_topic = tree.Sample(prob / beta);
      }
      else {
        prob -= prob_left;
        int p = (int)(lower_bound(psum.begin(), psum.begin() + items.size(), prob)
            - psum.begin());
        new_topic = items[p].item;
      }
      double accept = rand[thread].RandDouble();
      if (accept > exp((eta_n[new_topic] - eta_n[topic]) * (eta_z_o - eta_z) / sigma2)) {
        new_topic = topic;
      }

      word_dist.Inc(new_topic);
      word_dist.Unlock();

      doc_dist[new_topic]++;
      topics[i].topic = new_topic;
      topic_dist[new_topic]++;
      tree.Update(new_topic, CalDocTopic(new_topic, doc_dist[new_topic], topic_dist[new_topic],
                                     y - eta_z_o, eta_n[new_topic]));
      eta_z += eta_n[new_topic];
    }
  }
}

void SLdaModel::VisitByDoc(const Corpus &corpus) {
  static std::vector<int> doc_dist(n_topics), doc_dist_o;
  static std::vector<double> eta_n(n_topics);
#pragma omp parallel for schedule(dynamic) firstprivate(doc_dist) firstprivate(doc_dist_o) firstprivate(eta_n)
  for (int doc = 0; doc < corpus.n_words; ++doc) {
    int thread = omp_get_thread_num();
    int N = corpus.GetDocSize(doc);
    int offset = corpus.doc_offset[doc];

    eta_n.resize(n_topics);
    for (int k = 0; k < n_topics; ++k) {
      eta_n[k] = eta[k] / N;
    }
    doc_dist.clear();
    doc_dist.resize(n_topics);
    for (int i = 0; i < N; ++i) {
      doc_dist[topics[offset + i].topic]++;
    }

    double y_etaz = corpus.doc_infos[doc][0].double_value;
    for (int k = 0; k < n_topics; ++k) {
      y_etaz += eta_n[k] * doc_dist[k];
    }

    for (int i = 0; i < N; ++i) {
      if (!is_large_word[corpus.docs[offset + i]]) continue;
      int topic = topics[offset + i].topic;
      doc_dist[topic]--;
      topic_dist[topic]--;
      y_etaz -= eta_n[topic];

      for (int m = 0; m < MH_STEP; ++m) {
        int new_topic = topics[offset + i].mh_step[m];
        double Cwj = doc_dist[new_topic] + alpha;
        double Cwi = doc_dist[topic] + alpha;
        double Cj = topic_dist[new_topic] + corpus.n_words * beta;
        double Ci = topic_dist[topic] + corpus.n_words * beta;
        double ej = exp(eta_n[new_topic] * (2 * y_etaz - eta_n[new_topic]) / (2 * sigma2));
        double ei = exp(eta_n[topic] * (2 * y_etaz - eta_n[topic]) / (2 * sigma2));
        double prob = (Cwj * Ci) / (Cwi * Cj);
        if (rand[thread].RandDouble() < prob) {
          topic = new_topic;
        }
      }
      topic_dist[topic]++;
      doc_dist[topic]++;
      y_etaz += eta_n[topic];
      topics[offset + i].topic = topic;
    }

    double prob = (n_topics * beta) / (n_topics * beta + N);
    for (int i = 0; i < N; ++i) {
      if (!is_long_doc[corpus.words[offset + i]]) continue;
      for (int m = 0; m < MH_STEP; ++m) {
        if (rand[thread].RandDouble() < prob) {
          topics[offset + i].mh_step[m] = rand[thread].RandInt(n_topics);
        }
        else {
          topics[offset + i].mh_step[m] = topics[offset + rand[thread].RandInt(N)].topic;
        }
      }
    }
  }
}

void SLdaModel::VisitByDoc(const Corpus &corpus) {
  static std::vector<int> doc_dist;
  static std::vector<Token> tmp_token(n_topics);

#pragma omp parallel for schedule(dynamic) private(doc_dist) firstprivate(tmp_token)
  for (int doc = 0; doc < corpus.n_docs; ++doc) {
    if (!is_long_doc[doc]) continue;
    int thread = omp_get_thread_num();
    int N = corpus.doc_offset[doc + 1] - corpus.doc_offset[doc];
    int offset = corpus.doc_offset[doc];
    doc_dist.clear();
    doc_dist.resize(n_topics);
    tmp_token.clear();

    for (int i = 0; i < N; ++i) {
      tmp_token.push_back(topics[corpus.doc_to_word[offset + i]]);
      doc_dist[tmp_token[i].topic]++;
    }

    for (int i = 0; i < N; ++i) {
      Token& tok = tmp_token[i];
      int topic = tok.topic;
      doc_dist[topic]--;
      topic_dist[topic]--;

      for (int m = 0; m < MH_STEP; ++m) {
        int new_topic = tok.mh_step[m];
        double Cdj = doc_dist[new_topic] + alpha;
        double Cdi = doc_dist[topic] + alpha;
        double Cj = topic_dist[new_topic] + corpus.n_words * beta;
        double Ci = topic_dist[topic] + corpus.n_words * beta;
        double prob = (Cdj * Ci) / (Cdi * Cj);
        if (rand[thread].RandDouble() < prob) {
          topic = new_topic;
        }
      }
      topic_dist[topic]++;
      doc_dist[topic]++;
      tok.topic = topic;
    }

    double prob = (n_topics * alpha) / (n_topics * alpha + N);
    for (int i = 0; i < N; ++i) {
      Token& tok = tmp_token[i];
      for (int m = 0; m < MH_STEP; ++m) {
        if (rand[thread].RandDouble() < prob) {
          tok.mh_step[m] = rand[thread].RandInt(n_topics);
        }
        else {
          tok.mh_step[m] = tmp_token[rand[thread].RandInt(N)].topic;
        }
      }
    }
    for (int i = 0; i < N; ++i) {
      topics[corpus.doc_to_word[offset + i]] = tmp_token[i];
    }
  }
}

double SLdaModel::Loglikelihood(const Corpus &corpus) {
  double llh = 0;
  std::vector<int> doc_dist;
  for (int doc = 0; doc < corpus.n_docs; ++doc) {
    if (is_long_doc[doc]) {
      doc_dist.clear();
      doc_dist.resize(n_topics);
      for (int i = corpus.doc_offset[doc]; i < corpus.doc_offset[doc + 1]; ++i) {
        doc_dist[topics[corpus.doc_to_word[i]].topic]++;
      }
      for (int topic = 0; topic < n_topics; ++topic) {
        llh += lgamma(doc_dist[topic] + alpha);
      }
    }
    else {
      for (auto &item : doc_topic_dist[doc].GetItem()) {
        llh += lgamma(item.count + alpha);
      }
      llh += lgamma(alpha) * (n_topics - doc_topic_dist[doc].GetItem().size());
    }
  }
  for (int word = 0; word < corpus.n_words; ++word) {
    for (int topic = 0; topic < n_topics; ++topic) {
      llh += lgamma(word_topic_dist[word][topic] + beta);
    }
  }
  for (int topic = 0; topic < n_topics; ++topic) {
    llh -= lgamma(topic_dist[topic].load() + corpus.n_words * beta);
  }
  return llh;
}

void SLdaModel::Inference(const Corpus &corpus) {

}
void SLdaModel::LoadModel(const std::string &path) {

}
void SLdaModel::SaveModel(const std::string &path) {

}

}  // namespace liblda
 */
