import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
import random
import numpy as np

# Download NLTK data
nltk.download('punkt')

# Sample text corpus
corpus = """
The quick brown fox jumps over the lazy dog. The dog barked loudly.
The fox was clever and quick. The dog was lazy and sleepy.
"""

# Tokenize the text
tokens = word_tokenize(corpus.lower())

# Build bigrams and their frequencies
bigrams = list(nltk.bigrams(tokens))
bigram_freq = Counter(bigrams)

# Build a dictionary of probabilities for the next word
next_word_prob = defaultdict(lambda: defaultdict(int))
for (w1, w2), freq in bigram_freq.items():
    next_word_prob[w1][w2] = freq

# Convert counts to probabilities
for w1 in next_word_prob:
    total_count = float(sum(next_word_prob[w1].values()))
    for w2 in next_word_prob[w1]:
        next_word_prob[w1][w2] /= total_count

def generate_recursive_text(start_word, length, next_word_prob, generated=None):
    if generated is None:
        generated = [start_word]

    if length == 0:
        return ' '.join(generated)

    current_word = generated[-1]
    if current_word not in next_word_prob:
        return ' '.join(generated)  # End if no next word found

    next_word_candidates = list(next_word_prob[current_word].keys())
    next_word_probs = list(next_word_prob[current_word].values())
    next_word = np.random.choice(next_word_candidates, p=next_word_probs)

    generated.append(next_word)
    return generate_recursive_text(start_word, length - 1, next_word_prob, generated)

# Example usage
start_word = 'the'
generated_text = generate_recursive_text(start_word, 10, next_word_prob)
print("Generated Text:", generated_text)

