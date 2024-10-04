# Define the genres as a tuple array
genres = (
    'fantasy',
    'adventure',
    'mystical',
    'romance',
    'science',
    'action',
    'magical',
    'nonfiction',
    'fiction'
)

combined_genres = (
    ('Science Fiction',),
    ('Nonfiction Anime',),
    ('Fictional Science',),
    ('Magical Action',),
    ('Mystical Adventure',),
    ('Fantasy Adventure',),
    ('Magical Adventure',),
    ('Ghibli Anime',),
    ('Ghibli Cartoon',),
    ('Mystical Fantasy',),
    ('Mystical Action',),
    ('Mystical Romance',),
    ('Scientific Romance',),
    ('Magical Romance',),
    ('Fringe Based Romance',),
    ('Religious Romance',),
    ('Spiritual Romance',),
    ('Romance & Action',),
    ('Romance & Adventure',),
    ('Romantic Adventure',),
    ('Supernatural Romance',),
    ('Supernatural Adventure',),
    ('Mystical Fringe',),
    ('Magical Fringe',),
    ('Ghibli Fringe',),
    ('Ghibli Mystical',),
    ('Ghibli Magical',),
    ('Mystical Science',),
    ('Magical Science',),
    ('Fringe Science',),
    ('Magical Fantasy',),
    ('Mystical Fantasy',),
    ('Religious Fantasy',),
    ('Supernatural Fantasy',),
    ('Nonfiction Fantasy',),
    ('Fantasy Science',),
    ('Spiritual Fantasy',),
    ('Fringe Fantasy',),
    ('Fantasy Based Action',),
    ('Religious Suspense',),
    ('Religious Adventure',),
    ('Religious Fantasy',),
    ('Religious Science',),
    ('Religious Fiction',)
)


combined_genres = {
    'Science Fiction': ['Fantasy Science', 'Mystical Science', 'Magical Science'],
    'Nonfiction Anime': ['Ghibli Anime', 'Ghibli Cartoon'],
    'Fictional Science': ['Fantasy Science', 'Magical Science'],
    'Magical Action': ['Mystical Action', 'Romance & Action'],
    'Mystical Adventure': ['Fantasy Adventure', 'Magical Adventure'],
    'Fantasy Adventure': ['Mystical Fantasy', 'Magical Fantasy'],
    'Magical Adventure': ['Mystical Action', 'Romance & Adventure'],
    'Ghibli Anime': ['Nonfiction Anime', 'Ghibli Cartoon'],
    'Ghibli Cartoon': ['Nonfiction Anime', 'Ghibli Anime'],
    'Mystical Fantasy': ['Fantasy Adventure', 'Magical Fantasy'],
    'Mystical Action': ['Magical Action', 'Romance & Action'],
    'Mystical Romance': ['Romantic Adventure', 'Supernatural Romance'],
    'Scientific Romance': ['Magical Romance', 'Fringe Based Romance'],
    'Magical Romance': ['Mystical Romance', 'Romantic Adventure'],
    'Fringe Based Romance': ['Religious Romance', 'Spiritual Romance'],
    'Religious Romance': ['Supernatural Romance', 'Romantic Adventure'],
    'Spiritual Romance': ['Mystical Fantasy', 'Magical Fantasy'],
    'Romance & Action': ['Magical Action', 'Mystical Action'],
    'Romance & Adventure': ['Mystical Adventure', 'Magical Adventure'],
    'Romantic Adventure': ['Mystical Fantasy', 'Magical Fantasy'],
    'Supernatural Romance': ['Religious Romance', 'Spiritual Romance'],
    'Supernatural Adventure': ['Mystical Adventure', 'Magical Adventure'],
    'Mystical Fringe': ['Fringe Science', 'Fringe Fantasy'],
    'Magical Fringe': ['Fringe Science', 'Fringe Fantasy'],
    'Ghibli Fringe': ['Nonfiction Anime', 'Ghibli Cartoon'],
    'Ghibli Mystical': ['Mystical Fantasy', 'Magical Fantasy'],
    'Ghibli Magical': ['Magical Adventure', 'Mystical Adventure'],
    'Mystical Science': ['Fantasy Science', 'Magical Science'],
    'Magical Science': ['Fantasy Science', 'Mystical Science'],
    'Fringe Science': ['Mystical Fringe', 'Magical Fringe'],
    'Magical Fantasy': ['Fantasy Adventure', 'Mystical Adventure'],
    'Mystical Fantasy': ['Fantasy Adventure', 'Magical Adventure'],
    'Religious Fantasy': ['Supernatural Romance', 'Romantic Adventure'],
    'Nonfiction Fantasy': ['Fantasy Science', 'Mystical Science'],
    'Fantasy Science': ['Mystical Science', 'Magical Science'],
    'Spiritual Fantasy': ['Supernatural Romance', 'Romantic Adventure'],
    'Fringe Fantasy': ['Mystical Fringe', 'Magical Fringe'],
    'Fantasy Based Action': ['Magical Action', 'Mystical Action'],
    'Religious Suspense': ['Supernatural Romance', 'Romantic Adventure'],
    'Religious Adventure': ['Mystical Adventure', 'Magical Adventure'],
    'Religious Science': ['Mystical Science', 'Magical Science'],
    'Religious Fiction': ['Fantasy Fiction', 'Mystical Fiction']
}



import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the data
data = pd.read_csv('data.csv')

# Tokenize the text
data['text'] = data['text'].apply(word_tokenize)

# Remove stop words
stop_words = set(stopwords.words('english'))
data['text'] = data['text'].apply(lambda x: [word for word in x if word not in stop_words])

# Convert all text to lowercase
data['text'] = data['text'].apply(lambda x: [word.lower() for word in x])

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text'])