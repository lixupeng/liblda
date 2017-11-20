#include "model/tot.h"
#include "util/alias_table.h"
#include "util/ftree.h"
#include <omp.h>
#include <sys/time.h>

namespace liblda {

void TotModel::Train(const Corpus &corpus, int max_iter) {
  PrepareForTrain(corpus);
  struct timeval time_start, time_end;
  unsigned sample_time, total_time = 0;
  for (int i = 0; i < max_iter; ++i) {
    gettimeofday(&time_start, NULL);
    VisitByDoc(corpus);
    VisitByWord(corpus);
    FTreeIteration(corpus);
    EstimatePhi(corpus);
    gettimeofday(&time_end, NULL);
    sample_time = (unsigned int) ((time_end.tv_sec - time_start.tv_sec) * 1000000
        + time_end.tv_usec - time_start.tv_usec);
    total_time += sample_time;
    std::cout << "iter " << i << " " << sample_time << " " << total_time
              << "   " << "llh: " << Loglikelihood(corpus) << std::endl;
  }
}

void TotModel::PrepareForTrain(const Corpus &corpus) {
  topics.clear();       topics.resize(corpus.n_tokens);
  topic_dist.clear();   topic_dist.resize(n_topics);
  phi_a.clear();        phi_a.resize(n_topics);
  phi_b.clear();        phi_b.resize(n_topics);
  lgamma_phi.clear();   lgamma_phi.resize(n_topics);
  is_large_word.resize(corpus.n_words);

  doc_time.resize(corpus.n_docs);
  log_doc_time.resize(corpus.n_docs);
  log_1_doc_time.resize(corpus.n_docs);
  double min_time = 1e18, max_time = -1e18;
  for (int d = 0; d < corpus.n_docs; ++d) {
    min_time = std::min(min_time, GetDocTime(corpus, d));
    max_time = std::max(max_time, GetDocTime(corpus, d));
  }
  for (int d = 0; d < corpus.n_docs; ++d) {
    doc_time[d] = 0.1 + ((GetDocTime(corpus, d) - min_time)
        / (max_time - min_time + 1e-8)) * 0.8;
    log_doc_time[d] = log(doc_time[d] + 1e-16);
    log_1_doc_time[d] = log(1 - doc_time[d] + 1e-16);
  }

  AliasTable alias;
  std::vector<double> dist(n_topics);

  for (int w = 0; w < corpus.n_words; ++w) {
    std::string word = corpus.word_list[w];
    is_large_word[w] = (corpus.GetWordSize(w) > LARGE_WORD_SIZE);

    bool new_word = false;
    if (word_to_int.count(word) == 0) {
      word_to_int[word] = word_list.size();
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
  EstimatePhi(corpus);
}

void TotModel::FTreeIteration(const Corpus &corpus) {
  FTree tree(n_topics);
  std::vector<int> doc_dist;
  std::vector<double> psum;

#pragma omp parallel for schedule(dynamic) firstprivate(tree) private(psum) private(doc_dist)
  for (int doc = 0; doc < corpus.n_docs; ++doc) {
    int thread = omp_get_thread_num();

    doc_dist.clear();
    doc_dist.resize(n_topics);
    for (int i = corpus.doc_offset[doc]; i < corpus.doc_offset[doc + 1]; ++i) {
      doc_dist[topics[i].topic]++;
    }
    for (int k = 0; k < n_topics; ++k) {
      tree.Set(k, CalDocTopic(k, doc, doc_dist[k], topic_dist[k], corpus.n_words));
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
      tree.Update(topic, CalDocTopic(topic, doc, doc_dist[topic], topic_dist[topic], corpus.n_words));

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

      word_dist.Inc(new_topic);
      word_dist.Unlock();

      doc_dist[new_topic]++;
      topics[i].topic = new_topic;
      topic_dist[new_topic]++;
      tree.Update(new_topic, CalDocTopic(new_topic, doc, doc_dist[new_topic], topic_dist[new_topic], corpus.n_words));
    }
  }
}

void TotModel::VisitByDoc(const Corpus &corpus) {
  static std::vector<int> doc_dist(n_topics);
  static std::vector<double> prob(n_topics);
  AliasTable alias;
#pragma omp parallel for schedule(dynamic) firstprivate(doc_dist) firstprivate(prob) private(alias)
  for (int doc = 0; doc < corpus.n_docs; ++doc) {
    int thread = omp_get_thread_num();
    int N = corpus.GetDocSize(doc);
    int offset = corpus.doc_offset[doc];

    doc_dist.clear();
    doc_dist.resize(n_topics);
    for (int i = 0; i < N; ++i) {
      doc_dist[topics[offset + i].topic]++;
    }

    for (int i = 0; i < N; ++i) {
      if (!is_large_word[corpus.docs[offset + i]]) continue;
      int topic = topics[offset + i].topic;
      doc_dist[topic]--;
      topic_dist[topic]--;

      for (int m = 0; m < MH_STEP; ++m) {
        int new_topic = topics[offset + i].mh_step[m];
        double Bdj = exp(CalLogBetaTime(doc, new_topic));
        double Bdi = exp(CalLogBetaTime(doc, topic));
        double Cwj = doc_dist[new_topic] + alpha;
        double Cwi = doc_dist[topic] + alpha;
        double Cj = topic_dist[new_topic] + corpus.n_words * beta;
        double Ci = topic_dist[topic] + corpus.n_words * beta;
        double prob = (Bdj * Cwj * Ci) / (Bdi * Cwi * Cj);
        if (rand[thread].RandDouble() < prob) {
          topic = new_topic;
        }
      }
      topic_dist[topic]++;
      doc_dist[topic]++;
      topics[offset + i].topic = topic;
    }

    prob.resize(n_topics);
    for (int i = 0; i < n_topics; ++i) {
      prob[i] = CalDocTopic(i, doc, doc_dist[i], topic_dist[i], corpus.n_words);
    }
    alias.Init(prob);
    for (int i = 0; i < N; ++i) {
      if (!is_large_word[corpus.docs[offset + i]]) continue;
      for (int m = 0; m < MH_STEP; ++m) {
        topics[offset + i].mh_step[m] = alias.Sample(rand[thread]);
      }
    }
  }
}

void TotModel::VisitByWord(const Corpus &corpus) {
  static std::vector<int> word_dist;
  static std::vector<Token> tmp_token(n_topics);

#pragma omp parallel for schedule(dynamic) private(word_dist) firstprivate(tmp_token)
  for (int word = 0; word < corpus.n_words; ++word) {
    if (!is_large_word[word]) continue;
    int thread = omp_get_thread_num();
    int N = corpus.GetWordSize(word);
    int offset = corpus.word_offset[word];
    word_dist.clear();
    word_dist.resize(n_topics);
    tmp_token.clear();

    for (int i = 0; i < N; ++i) {
      tmp_token.push_back(topics[corpus.word_to_doc[offset + i]]);
      word_dist[tmp_token[i].topic]++;
    }

    for (int i = 0; i < N; ++i) {
      Token& tok = tmp_token[i];
      int topic = tok.topic;
      word_dist[topic]--;
      topic_dist[topic]--;

      for (int m = 0; m < MH_STEP; ++m) {
        int new_topic = tok.mh_step[m];
        double Cwj = word_dist[new_topic] + beta;
        double Cwi = word_dist[topic] + beta;
        double prob = Cwj / Cwi;
        if (rand[thread].RandDouble() < prob) {
          topic = new_topic;
        }
      }
      topic_dist[topic]++;
      word_dist[topic]++;
      tok.topic = topic;
    }

    double prob = (n_topics * beta) / (n_topics * beta + N);
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
      topics[corpus.word_to_doc[offset + i]] = tmp_token[i];
    }
  }
}

void TotModel::EstimatePhi(const Corpus &corpus) {
  static std::vector<double> mean, var;
  mean.clear();
  var.clear();
  mean.resize(n_topics);
  var.resize(n_topics);
  for (int doc = 0; doc < corpus.n_docs; ++doc) {
    for (int t = corpus.doc_offset[doc]; t < corpus.doc_offset[doc + 1]; ++t) {
      mean[topics[t].topic] += doc_time[doc];
    }
  }
  for (int i = 0; i < n_topics; ++i) {
    if (topic_dist[i] > 0) {
      mean[i] /= topic_dist[i];
    }
  }
    for (int doc = 0; doc < corpus.n_docs; ++doc) {
      for (int t = corpus.doc_offset[doc]; t < corpus.doc_offset[doc + 1]; ++t) {
        double diff = mean[topics[t].topic] - doc_time[doc];
        var[topics[t].topic] += diff * diff;
      }
  }
  for (int i = 0; i < n_topics; ++i) {
    if (topic_dist[i] > 1) {
      var[i] /= (topic_dist[i] - 1);
    }
  }
  for (int i = 0; i < n_topics; ++i) {
    if (var[i] > mean[i] * (1 - mean[i]) - 1e-5) {
      phi_a[i] = (mean[i] + 1e-5) * 0.1;
      phi_b[i] = (1 - mean[i] + 1e-5) * 0.1;
      lgamma_phi[i] = lgamma(phi_a[i]) + lgamma(phi_b[i]) - lgamma(phi_a[i] + phi_b[i]);
      continue;
    }
    phi_a[i] = mean[i] * (mean[i] * (1 - mean[i]) / (var[i] + 1e-5) - 1);
    phi_b[i] = (1 - mean[i]) * (mean[i] * (1 - mean[i]) / (var[i] + 1e-5) - 1);
    lgamma_phi[i] = lgamma(phi_a[i]) + lgamma(phi_b[i]) - lgamma(phi_a[i] + phi_b[i]);
  }
}

double TotModel::Loglikelihood(const Corpus &corpus) {
  double llh = 0;
  std::vector<int> dist;
  for (int doc = 0; doc < corpus.n_docs; ++doc) {
    dist.clear();
    dist.resize(n_topics);
    for (int i = corpus.doc_offset[doc]; i < corpus.doc_offset[doc + 1]; ++i) {
      dist[topics[i].topic]++;
    }
    for (int topic = 0; topic < n_topics; ++topic) {
      llh += lgamma(dist[topic] + alpha);
      llh += dist[topic] * CalLogBetaTime(doc, topic);
    }
  }
  for (int word = 0; word < corpus.n_words; ++word) {
    if (is_large_word[word]) {
      for (int topic = 0; topic < n_topics; ++topic) {
        llh += lgamma(word_topic_dist[word].Count(topic) + beta);
      }
    }
    else {
      dist.clear();
      dist.resize(n_topics);
      for (int i = corpus.word_offset[word]; i < corpus.word_offset[word + 1]; ++i) {
        dist[topics[corpus.word_to_doc[i]].topic]++;
      }
      for (int topic = 0; topic < n_topics; ++topic) {
        llh += lgamma(dist[topic] + beta);
      }
    }
  }
  for (int topic = 0; topic < n_topics; ++topic) {
    llh -= lgamma(topic_dist[topic].load() + corpus.n_words * beta);
  }
  return llh;
}

void TotModel::Inference(const Corpus &corpus) {

}
void TotModel::LoadModel(const std::string &path) {

}
void TotModel::SaveModel(const std::string &path) {

}

}  // namespace liblda
