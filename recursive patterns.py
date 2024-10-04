import re
from collections import defaultdict

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    return words

# Sample text
text = """
The quick brown fox jumps over the lazy dog. The dog barked loudly.
The fox was clever and quick. The dog was lazy and sleepy.
"""

words = preprocess_text(text)
print("Words:", words)

def recursive_pattern_recognition(words, min_length=1, max_length=3, depth=0):
    if len(words) < min_length:
        return defaultdict(int)

    patterns = defaultdict(int)
    for length in range(min_length, max_length + 1):
        for i in range(len(words) - length + 1):
            pattern = tuple(words[i:i + length])
            patterns[pattern] += 1

    if depth < max_length:
        mid = len(words) // 2
        left_patterns = recursive_pattern_recognition(words[:mid], min_length, max_length, depth + 1)
        right_patterns = recursive_pattern_recognition(words[mid:], min_length, max_length, depth + 1)
        
        for pattern, count in left_patterns.items():
            patterns[pattern] += count
        for pattern, count in right_patterns.items():
            patterns[pattern] += count

    return patterns

# Example usage
patterns = recursive_pattern_recognition(words)
sorted_patterns = sorted(patterns.items(), key=lambda x: (-x[1], x[0]))
for pattern, count in sorted_patterns:
    print(f"Pattern: {' '.join(pattern)}, Count: {count}")
