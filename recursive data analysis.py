interview_transcript = """
Interviewer: Can you describe your experience with the project?
Interviewee: Yes, I worked on the project for six months. During this time, I was responsible for managing the team and ensuring that all deadlines were met.
Interviewer: What challenges did you face?
Interviewee: One of the main challenges was coordinating between different departments. Communication was key to overcoming this.
Interviewer: How did you resolve these challenges?
Interviewee: We held regular meetings and used project management tools to keep everyone on the same page. This helped us to stay organized and address issues quickly.
"""

# Function to clean and split text into words
import re
from collections import Counter

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def split_into_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def split_into_words(sentence):
    words = sentence.split()
    return words

def recursive_word_count(text, depth=0, max_depth=2):
    if depth >= max_depth:
        words = split_into_words(clean_text(text))
        return Counter(words)
    
    sentences = split_into_sentences(text)
    overall_counter = Counter()
    
    for sentence in sentences:
        overall_counter += recursive_word_count(sentence, depth + 1, max_depth)
    
    return overall_counter

# Perform the recursive data analysis
word_count = recursive_word_count(interview_transcript)

# Display the most common words
most_common_words = word_count.most_common(10)
print("Most common words:", most_common_words)
