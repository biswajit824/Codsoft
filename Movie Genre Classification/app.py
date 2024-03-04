import streamlit as st
import pandas as pd
import pickle
import re
import string

# Load the trained model
with open("movie_genre_classification.pkl", "rb") as file:
    model = pickle.load(file)

# Create a function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

import numpy as np

# # Create a function to predict genres
# def predict_genre(plot_summary):
#     plot_summary_processed = preprocess_text(plot_summary)
#     # Reshape the input data to a 2D array
#     plot_summary_2d = np.array([plot_summary_processed]).reshape(-1, 1)
#     predicted_genres = model.predict(plot_summary_2d)
#     return predicted_genres[0]


from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf_vectorizer = pickle.load(file)

# Create a function to predict genres
def predict_genre(plot_summary):
    plot_summary_processed = preprocess_text(plot_summary)
    # Vectorize the input data using TF-IDF
    plot_summary_vectorized = tfidf_vectorizer.transform([plot_summary_processed])
    predicted_genres = model.predict(plot_summary_vectorized)
    return predicted_genres[0]




# Streamlit UI
st.title('Movie Genre Classification')

# Input for plot summary
plot_summary = st.text_area('Enter the plot summary of the movie:')

# Button to classify genre
if st.button('Classify Genre'):
    if plot_summary:
        # Make prediction
        genre = predict_genre(plot_summary)
        st.success(f'The predicted genre(s) for the movie is/are: {genre}')
    else:
        st.warning('Please enter a plot summary.')