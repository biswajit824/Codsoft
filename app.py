# import streamlit as st
# import pickle
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# with open("movie_genre_classification.pkl", "rb") as file:
#     model = pickle.load(file)
# # model = pickle.load('movie_genre_classification_.pkl')
# # vectorizer = pickle.load('tfidf_vectorizer.pkl')
# vectorizer = pickle.load('tfidf_vectorizer.pkl')

# # Create a function to predict genres
# def predict_genre(plot_summary):
#     # Vectorize the input plot summary
#     plot_summary_vectorized = vectorizer.transform([plot_summary])
#     # Predict the genres
#     predicted_genres = model.predict(plot_summary_vectorized)
#     return predicted_genres[0]



# st.title("Movie Genre Classification")
# plot_summary = st.text_area('Enter the plot summary of the movie:')
# if st.button('Classify Genre'):
#     if plot_summary:
#         # Make prediction
#         genre = predict_genre(plot_summary)
#         st.success(f'The predicted genre(s) for the movie is/are: {genre}')
#     else:
#         st.warning('Please enter a plot summary.')


# # Sample plot summary
# st.sidebar.header('Sample Plot Summary')
# sample_plot_summary = """
# A young boy named Andy loves to be in his room, playing with his toys, especially his doll named "Woody". 
# But, what do the toys do when Andy is not with them, they come to life. Woody believes that he has life 
# (as a toy) good. However, he must worry about Andy's family moving, and what Woody does not know is about 
# Andy's birthday party. Woody does not realize that Andy's mother gave him an action figure known as Buzz Lightyear, 
# who does not believe that he is a toy, and quickly becomes Andy's new favorite toy. Woody, who is now consumed 
# with jealousy, tries to get rid of Buzz. Then, both Woody and Buzz are now lost. They must find a way to get back to 
# Andy before he moves without them, but they will have to pass through a ruthless toy killer, Sid Phillips.
# """
# st.sidebar.text(sample_plot_summary)       







import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import re
import string

# Load the trained model

with open("movie_genre_classification.pkl", "rb") as file:
       model = pickle.load(file)

# Create a function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create a function to predict genres
def predict_genre(plot_summary):
    # Preprocess the input plot summary
    plot_summary_processed = preprocess_text(plot_summary)
    # Predict the genres
    predicted_genres = model.predict([plot_summary_processed])
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

# Sample plot summary
st.sidebar.header('Sample Plot Summary')
sample_plot_summary = """
A young boy named Andy loves to be in his room, playing with his toys, especially his doll named "Woody". 
But, what do the toys do when Andy is not with them, they come to life. Woody believes that he has life 
(as a toy) good. However, he must worry about Andy's family moving, and what Woody does not know is about 
Andy's birthday party. Woody does not realize that Andy's mother gave him an action figure known as Buzz Lightyear, 
who does not believe that he is a toy, and quickly becomes Andy's new favorite toy. Woody, who is now consumed 
with jealousy, tries to get rid of Buzz. Then, both Woody and Buzz are now lost. They must find a way to get back to 
Andy before he moves without them, but they will have to pass through a ruthless toy killer, Sid Phillips.
"""
st.sidebar.text(sample_plot_summary)
