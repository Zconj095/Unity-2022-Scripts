import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Sample documents
documents = [
    "Machine learning is a method of data analysis that automates analytical model building.",
    "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
    "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."
]

# Tokenize and preprocess the text
nltk.download('punkt')
tokenized_documents = [nltk.word_tokenize(doc.lower()) for doc in documents]

# Function to preprocess text
def preprocess_text(doc):
    # Tokenization
    tokens = nltk.word_tokenize(doc.lower())
    # Remove non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]
    return ' '.join(tokens)

# Preprocess all documents
processed_documents = [preprocess_text(doc) for doc in documents]
print("Processed Documents:", processed_documents)

# Vectorize the documents using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_documents)

# Train a Nearest Neighbors model
model = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X)

# Function to query the model
def query_model(query):
    query_processed = preprocess_text(query)
    query_vector = vectorizer.transform([query_processed])
    distance, index = model.kneighbors(query_vector)
    return documents[index[0][0]], distance[0][0]

# Test the model with a new query
query = "What is deep learning?"
result, distance = query_model(query)
print(f"Query: {query}\nResult: {result}\nDistance: {distance}")
