#include "corpus/corpus.h"
#include "util/random.h"

namespace liblda {

void Corpus::ReadFromFile(const std::string &doc_path) {
  std::ifstream fin(doc_path);
  std::string doc;

  int read_buffer_size = 1 << 20;
  char read_buffer[read_buffer_size];
  std::vector<int> words(256);

  // read documents
  while (fin >> doc) {
    // read words
    fin.getline(read_buffer, read_buffer_size, '\n');
    std::stringstream doc_s(read_buffer);

    words.clear();

    std::string word;
    while (doc_s >> word) {
      int word_id = GetWordId(word);
      words.push_back(word_id);
    }
    AddDoc(doc, words);
  }
  fin.close();

  FinishAdding();
}

void Corpus::AddDoc(const std::string &doc, const std::vector<int> &words,
                    const std::vector<Value> &doc_info) {
  if (doc_to_int.count(doc) == 0) {
    GetDocId(doc);
    doc_offset.push_back(docs.size());
    docs.insert(docs.end(), words.begin(), words.end());
    doc_infos.push_back(doc_info);
  }
}

void Corpus::FinishAdding() {
  n_tokens = docs.size();
  words.reserve(n_tokens);
  doc_offset.push_back(docs.size());
  word_offset.resize(n_words + 1);
  doc_to_word.reserve(n_tokens);
  word_to_doc.reserve(n_tokens);

  for (int i = 0; i < n_tokens; ++i)
    word_offset[docs[i]]++;
  for (int i = 1; i < n_words; ++i)
    word_offset[i] += word_offset[i - 1];
  word_offset[n_words] = word_offset[n_words - 1];
  for (int d = 0; d < n_docs; ++d) {
    for (int i = doc_offset[d]; i < doc_offset[d + 1]; ++i) {
      int w = docs[i];
      words[--word_offset[w]] = d;
      doc_to_word[i] = word_offset[w];
      word_to_doc[word_offset[w]] = i;
    }
  }

  // shuffle words_
  for (int word = 0; word < n_words; ++word) {
    int begin = word_offset[word];
    int end = word_offset[word + 1];
    int N = end - begin;
    Random rand;
    for (int i = 0; i < 2 * N; ++i) {
      int a = rand.RandInt(N) + begin;
      int b = rand.RandInt(N) + begin;
      std::swap(words[a], words[b]);
      std::swap(word_to_doc[a], word_to_doc[b]);
      doc_to_word[word_to_doc[a]] = a;
      doc_to_word[word_to_doc[b]] = b;
    }
  }

  finished = true;
}

int Corpus::GetDocId(const std::string &doc) {
  if (doc_to_int.count(doc) == 0) {
    doc_list.push_back(doc);
    doc_to_int[doc] = n_docs;
    return n_docs++;
  }
  else {
    return doc_to_int[doc];
  }
}

int Corpus::GetWordId(const std::string &word) {
  if (word_to_int.count(word) == 0) {
    word_list.push_back(word);
    word_to_int[word] = n_words;
    return n_words++;
  }
  else {
    return word_to_int[word];
  }
}

}  // namespace liblda