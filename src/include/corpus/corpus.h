#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <iostream>
#include <mutex>
#include <cmath>

namespace liblda {

enum class ValueType {
  INTEGER,
  DOUBLE,
};

struct Value {
  ValueType type;
  union {
    int int_value;
    double double_value;
  };
  Value(int val) {
    type = ValueType::INTEGER;
    int_value = val;
  }
  Value(double val) {
    type = ValueType::DOUBLE;
    double_value = val;
  }
};

struct Corpus {
  void ReadFromFile(const std::string &doc_path);

  void AddDoc(const std::string &doc, const std::vector<int> &word,
              const std::vector<Value> &doc_info = {});
  void FinishAdding();

  bool IsFinished() const { return finished; }

  std::string GetWordById(int id) const {
    return word_list[id];
  }
  int GetWordSize(int id) const {
    return word_offset[id + 1] - word_offset[id];
  }
  int GetDocSize(int id) const {
    return doc_offset[id + 1] - doc_offset[id];
  }

  // Number of documents
  int n_docs = 0;
  // Number of vocabulary
  int n_words = 0;
  // Number of tokens
  int n_tokens = 0;
  // Document Ids organized by word
  std::vector<int> words;
  // Word Ids organized by document
  std::vector<int> docs;
  // index map from docs to words
  std::vector<int> doc_to_word;
  // index map from words to docs
  std::vector<int> word_to_doc;
  // Cumulative counts for the number of words
  std::vector<int> word_offset;
  // Cumulative counts for the number of documents
  std::vector<int> doc_offset;
  // Vocabulary map
  std::unordered_map<std::string, int> word_to_int;
  // Document map
  std::unordered_map<std::string, int> doc_to_int;
  // Vocabulary list
  std::vector<std::string> word_list;
  // Document list
  std::vector<std::string> doc_list;
  // Document infos
  std::vector<std::vector<Value>> doc_infos;

  // get doc id
  int GetDocId(const std::string &doc);

  int GetWordId(const std::string &word);

  bool finished = false;
};

}  // namespace liblda
