import re
# Sample interview transcripts
transcripts = [
    "I think remote learning is effective because it allows flexibility in scheduling.",
    "Remote learning can be challenging due to lack of face-to-face interaction.",
    "The effectiveness of remote learning depends on the course material and instructor.",
    "I find remote learning to be less engaging compared to in-person classes.",
]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

# Preprocess all transcripts
preprocessed_transcripts = [preprocess_text(transcript) for transcript in transcripts]

from collections import defaultdict, Counter

# Coding the data by identifying key themes
def code_data(transcripts):
    codes = []
    for transcript in transcripts:
        if "effective" in transcript or "effectiveness" in transcript:
            codes.append("effectiveness")
        if "flexibility" in transcript:
            codes.append("flexibility")
        if "challenging" in transcript:
            codes.append("challenges")
        if "interaction" in transcript:
            codes.append("interaction")
        if "engaging" in transcript:
            codes.append("engagement")
    return codes

# Identify themes from coded data
def identify_themes(codes):
    theme_counts = Counter(codes)
    themes = {theme: count for theme, count in theme_counts.items()}
    return themes

# Perform initial analysis
codes = code_data(preprocessed_transcripts)
themes = identify_themes(codes)
print("Initial Themes:", themes)

# Simulated additional data based on refined research question
additional_transcripts = [
    "The lack of face-to-face interaction makes it hard to stay engaged in remote learning.",
    "Engagement in remote learning can be improved with interactive activities.",
    "Face-to-face interaction is crucial for maintaining high levels of student engagement.",
]

# Preprocess the new transcripts
additional_preprocessed = [preprocess_text(transcript) for transcript in additional_transcripts]

# Combine with initial data
all_transcripts = preprocessed_transcripts + additional_preprocessed

# Re-analyze with the combined data
codes = code_data(all_transcripts)
themes = identify_themes(codes)
print("Refined Themes:", themes)

# Further refine research question if needed
# New insights might focus on specific strategies to improve engagement
