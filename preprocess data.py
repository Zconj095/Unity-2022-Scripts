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
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Preprocess all transcripts
preprocessed_transcripts = [preprocess_text(transcript) for transcript in transcripts]

def code_data(transcripts):
    codes = []
    for transcript in transcripts:
        if "successful" in transcript:
            codes.append("success")
        if "challenges" in transcript:
            codes.append("challenges")
        if "rewarding" in transcript:
            codes.append("rewarding")
        if "setbacks" in transcript:
            codes.append("setbacks")
        if "dedication" in transcript:
            codes.append("dedication")
    return codes

# Code the data
codes = code_data(preprocessed_transcripts)
print("Codes:", codes)

def identify_themes(codes):
    theme_counts = Counter(codes)
    themes = {theme: count for theme, count in theme_counts.items() if count > 1}
    return themes

# Identify themes
themes = identify_themes(codes)
print("Themes:", themes)

def develop_narrative(themes):
    narrative = []
    if "success" in themes:
        narrative.append("The project was generally viewed as successful.")
    if "challenges" in themes:
        narrative.append("There were challenges faced during the project.")
    if "rewarding" in themes:
        narrative.append("The experience was rewarding for the team.")
    if "setbacks" in themes:
        narrative.append("The team encountered some setbacks.")
    if "dedication" in themes:
        narrative.append("The team's dedication was crucial to overcoming obstacles.")
    return " ".join(narrative)

# Develop the narrative
narrative = develop_narrative(themes)
print("Narrative:", narrative)

def recursive_transform(transcripts, iterations=3):
    for i in range(iterations):
        print(f"Iteration {i+1}")
        codes = code_data(transcripts)
        themes = identify_themes(codes)
        narrative = develop_narrative(themes)
        print("Codes:", codes)
        print("Themes:", themes)
        print("Narrative:", narrative)
        print("-" * 40)

# Perform recursive transformation
recursive_transform(preprocessed_transcripts)
