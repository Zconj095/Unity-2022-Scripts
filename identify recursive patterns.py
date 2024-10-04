import re
from collections import defaultdict, Counter

# Sample interview transcripts
transcripts = [
    "I think the project was very successful. The team worked really hard, and we met all our deadlines.",
    "The project had some challenges, but overall, it was a great learning experience for everyone involved.",
    "Working on the project was a rewarding experience. I learned a lot, and we delivered a high-quality product.",
    "There were some setbacks during the project, but the team's dedication helped us overcome them.",
]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return words

# Preprocess all transcripts
preprocessed_transcripts = [preprocess_text(transcript) for transcript in transcripts]

def recursive_pattern_recognition(data, min_length=1, max_length=3, depth=0):
    if len(data) < min_length:
        return defaultdict(int)

    patterns = defaultdict(int)
    for length in range(min_length, max_length + 1):
        for segment in data:
            for i in range(len(segment) - length + 1):
                pattern = tuple(segment[i:i + length])
                patterns[pattern] += 1

    if depth < max_length:
        mid = len(data) // 2
        left_patterns = recursive_pattern_recognition(data[:mid], min_length, max_length, depth + 1)
        right_patterns = recursive_pattern_recognition(data[mid:], min_length, max_length, depth + 1)

        for pattern, count in left_patterns.items():
            patterns[pattern] += count
        for pattern, count in right_patterns.items():
            patterns[pattern] += count

    return patterns

# Identify patterns in the preprocessed transcripts
patterns = recursive_pattern_recognition(preprocessed_transcripts)
sorted_patterns = sorted(patterns.items(), key=lambda x: (-x[1], x[0]))

def identify_trends(patterns):
    trends = defaultdict(list)
    for pattern, count in patterns.items():
        trends[len(pattern)].append((pattern, count))
    return trends

trends = identify_trends(patterns)

def interpret_results(sorted_patterns):
    print("Top patterns and their counts:")
    for pattern, count in sorted_patterns[:10]:  # Display top 10 patterns
        print(f"Pattern: {' '.join(pattern)}, Count: {count}")

    print("\nTrends based on pattern lengths:")
    for length, items in trends.items():
        print(f"Pattern length {length}:")
        for pattern, count in sorted(items, key=lambda x: -x[1]):
            print(f"  {' '.join(pattern)}, Count: {count}")

# Interpret the results
interpret_results(sorted_patterns)
    