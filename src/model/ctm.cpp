#include "model/ctm.h"
#include "util/alias_table.h"
#include "util/ftree.h"
#include "util/numeric/logdet.h"
#include <omp.h>
#include <sys/time.h>

namespace liblda {

void CtmModel::Train(const Corpus &corpus, int max_iter) {
  PrepareForTrain(corpus);
  struct timeval time_start, time_end;
  unsigned sample_time, total_time = 0;
  for (int i = 0; i < max_iter; ++i) {
    gettimeofday(&time_start, NULL);
    VisitByDoc(corpus);
    VisitByWord(corpus);
    FTreeIteration(corpus);
    SampleOthers(corpus);
    gettimeofday(&time_end, NULL);
    sample_time = (unsigned int) ((time_end.tv_sec - time_start.tv_sec) * 1000000 + time_end.tv_usec - time_start.tv_usec);
    total_time += sample_time;
    std::cout << "iter " << i << " " << sample_time << " " << total_time
              << "   llh: " << Loglikelihood(corpus) << std::endl;
  }
}

void CtmModel::PrepareForTrain(const Corpus &corpus) {
  topics.clear();       topics.resize(corpus.n_tokens);
  topic_dist.clear();   topic_dist.resize(n_topics);
  is_large_word.resize(corpus.n_words);

  eta.resize(corpus.n_docs, n_topics);
  exp_eta.resize(corpus.n_docs, n_topics);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < corpus.n_docs; ++i) {
    int thread = omp_get_thread_num();
    for (int j = 0; j < n_topics; ++j) {
      eta(i, j) = rand[thread].RandNorm(mu_, sigma_);
      exp_eta(i, j) = exp(eta(i, j));
    }
  }
  mu.resize(n_topics);
  mu.fill(mu_);
  sigma.resize(n_topics, n_topics);
  for (int i = 0; i < n_topics; ++i) {
    sigma(i, i) = sigma_;
  }
  inv_sigma = sigma.inverse();

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
}

void CtmModel::FTreeIteration(const Corpus &corpus) {
  FTree tree(n_topics);
  std::vector<double> psum;

#pragma omp parallel for schedule(dynamic) firstprivate(tree) private(psum)
  for (int doc = 0; doc < corpus.n_docs; ++doc) {
    int thread = omp_get_thread_num();

    for (int k = 0; k < n_topics; ++k) {
      tree.Set(k, CalDocTopic(k, topic_dist[k], exp_eta(doc, k), corpus.n_words));
    }
    tree.Build();
    for (int i = corpus.doc_offset[doc]; i < corpus.doc_offset[doc]; ++i) {
      int word = corpus.docs[i];
      if (is_large_word[word]) continue;
      int topic = topics[i].topic;
      SparseCounter &word_dist = word_topic_dist[doc];
      word_dist.Lock();

      word_dist.Dec(topic);
      topic_dist[topic]--;
      tree.Update(topic, CalDocTopic(topic, topic_dist[topic],
                                     exp_eta(doc, topic), corpus.n_words));

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

      topics[i].topic = new_topic;
      topic_dist[new_topic]++;
      tree.Update(new_topic, CalDocTopic(new_topic, topic_dist[new_topic],
                                         exp_eta(doc, new_topic), corpus.n_words));
    }
  }
}

void CtmModel::VisitByDoc(const Corpus &corpus) {
  std::vector<double> prob;
  AliasTable alias;
#pragma omp parallel for schedule(dynamic) private(alias), private(prob)
  for (int doc = 0; doc < corpus.n_docs; ++doc) {
    int thread = omp_get_thread_num();
    int N = corpus.GetDocSize(doc);
    int offset = corpus.doc_offset[doc];

    for (int i = 0; i < N; ++i) {
      if (!is_large_word[corpus.docs[offset + i]]) continue;
      int topic = topics[offset + i].topic;
      topic_dist[topic]--;

      for (int m = 0; m < MH_STEP; ++m) {
        int new_topic = topics[offset + i].mh_step[m];
        double Cj = topic_dist[new_topic] + corpus.n_words * beta;
        double Ci = topic_dist[topic] + corpus.n_words * beta;
        double ej = exp_eta(doc, new_topic);
        double ei = exp_eta(doc, topic);
        double prob = (ej * Ci) / (ei * Cj);
        if (rand[thread].RandDouble() < prob) {
          topic = new_topic;
        }
      }
      topic_dist[topic]++;
      topics[offset + i].topic = topic;
    }

    prob.resize(n_topics);
    for (int i = 0; i < n_topics; ++i) {
      prob[i] = exp_eta(doc, i) / (topic_dist[i] + corpus.n_words * beta);
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

void CtmModel::VisitByWord(const Corpus &corpus) {
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

void CtmModel::SampleOthers(const Corpus &corpus) {
  Eigen::VectorXd delta(n_topics);
  for (int iter = 0; iter < SGLD_ITER_NUM; ++iter) {
    double eps = 0.5 * pow(100.0 + iter, -0.75);
#pragma omp parallel for schedule(dynamic) firstprivate(delta)
    for (int doc = 0; doc < corpus.n_docs; ++doc) {
      int thread = omp_get_thread_num();
      delta = -(eta.row(doc) - mu.transpose()) * inv_sigma;
      int Nd = corpus.GetDocSize(doc);
      for (int i = corpus.doc_offset[doc]; i < corpus.doc_offset[doc + 1]; ++i) {
        delta(topics[i].topic) += 1;
      }
      double sum_exp_eta = exp_eta.row(doc).sum();
      for (int k = 0; k < n_topics; ++k) {
        delta(k) -= Nd * eta(doc, k) / sum_exp_eta;
      }
      delta *= 0.5 * eps;
      for (int k = 0; k < n_topics; ++k) {
        delta(k) += rand[thread].RandNorm(0.0, eps);
      }
      eta.row(doc) += delta;
    }
  }
  mu.fill(0);
  for (int d = 0; d < corpus.n_docs; ++d) {
    mu += eta.row(d);
  }
  mu /= n_topics;
  sigma.fill(0);
  for (int d = 0; d < corpus.n_docs; ++d) {
    sigma += (eta.row(d).transpose() - mu) * (eta.row(d) - mu.transpose());
  }
  sigma /= n_topics;
  inv_sigma = sigma.inverse();
  for (int d = 0; d < corpus.n_docs; ++d) {
    for (int i = 0; i < n_topics; ++i) {
      exp_eta(d, i) = exp(eta(d, i));
    }
  }
}

double CtmModel::Loglikelihood(const Corpus &corpus) {
  double llh = 0;
  std::vector<int> dist;
  double ldt = logdet(sigma);
  for (int doc = 0; doc < corpus.n_docs; ++doc) {
    dist.clear();
    dist.resize(n_topics);
    for (int i = corpus.doc_offset[doc]; i < corpus.doc_offset[doc + 1]; ++i) {
      dist[topics[i].topic]++;
    }
    for (int topic = 0; topic < n_topics; ++topic) {
      llh += dist[topic] * eta(doc, topic);
    }
    double sum_exp_eta = exp_eta.row(doc).sum();
    llh -= corpus.GetDocSize(doc) * log(sum_exp_eta);
    llh += -0.5 * ldt -0.5 * (eta.row(doc) - mu.transpose()) * inv_sigma *
        (eta.row(doc).transpose() - mu);
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
}

void CtmModel::Inference(const Corpus &corpus) {

}
void CtmModel::LoadModel(const std::string &path) {

}
void CtmModel::SaveModel(const std::string &path) {

}

}  // namespace liblda
