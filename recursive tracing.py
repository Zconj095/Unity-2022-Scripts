import re
from collections import defaultdict, Counter
import numpy as np
import networkx as nx

# Sample text corpus
text_corpus = """
Recursive tracing is a method for collecting statistics about a language model.
It works by recursively following the links between words in the model's vocabulary.
This allows the statistics to be collected for any given word, regardless of its position in the model.
Recursive tracing can be used to collect a variety of statistics about a language model.
These statistics can be used to improve the model's performance, or to better understand how the model works.
"""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    return words

# Preprocess the text corpus
words = preprocess_text(text_corpus)
print("Words:", words)

# Collect word frequency
def collect_word_frequency(words):
    frequency = Counter(words)
    return frequency

# Collect word co-occurrence
def collect_word_cooccurrence(words, window_size=2):
    cooccurrence = defaultdict(Counter)
    for i, word in enumerate(words):
        for j in range(1, window_size + 1):
            if i + j < len(words):
                cooccurrence[word][words[i + j]] += 1
            if i - j >= 0:
                cooccurrence[word][words[i - j]] += 1
    return cooccurrence

# Compute word similarity based on co-occurrence
def compute_word_similarity(cooccurrence):
    similarity = defaultdict(dict)
    for word1 in cooccurrence:
        for word2 in cooccurrence[word1]:
            similarity[word1][word2] = cooccurrence[word1][word2]
    return similarity

# Perform recursive tracing
word_frequency = collect_word_frequency(words)
word_cooccurrence = collect_word_cooccurrence(words)
word_similarity = compute_word_similarity(word_cooccurrence)

print("Word Frequency:", word_frequency)
print("Word Co-occurrence:", word_cooccurrence)
print("Word Similarity:", word_similarity)

