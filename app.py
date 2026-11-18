
import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.metrics.pairwise import cosine_similarity


# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)


# Text cleaning function
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        word for word in words
        if word not in ENGLISH_STOP_WORDS
    ]

    return " ".join(words)


# Load dataset and build TF-IDF vectors
@st.cache_resource
def load_data():

    df = pd.read_csv("tmdb_5000_movies.csv")

    # Check required columns
    if "title" not in df.columns or "overview" not in df.columns:
        raise ValueError(
            "Dataset must contain title and overview columns."
        )

    # Handle missing values
    df["title"] = df["title"].fillna("").astype(str)
    df["overview"] = df["overview"].fillna("").astype(str)

    # Remove rows without movie titles
    df = df[df["title"].str.strip() != ""]
    df = df.reset_index(drop=True)

    # Clean text
    df["clean_text"] = df["overview"].apply(clean_text)

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(df["clean_text"])

    return df, tfidf_matrix


# Recommendation function
def recommend(movie_name, df, tfidf_matrix, top_n=5):

    movie_index = df[
        df["title"] == movie_name
    ].index[0]

    # Calculate similarity for the selected movie
    scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Sort scores
    similar_indices = np.argsort(scores)[::-1]

    recommendations = []

    for index in similar_indices:

        if index == movie_index:
            continue

        recommendations.append({
            "Movie": df.iloc[index]["title"],
            "Similarity Score": round(
                float(scores[index]), 3
            )
        })

        if len(recommendations) == top_n:
            break

    return recommendations


# Main application
st.title("🎬 Movie Recommendation System")

st.write(
    "Select your favourite movie to discover "
    "similar movies."
)

try:

    df, tfidf_matrix = load_data()

    st.success("Movie dataset loaded successfully!")

    # Movie selection dropdown
    movie_list = df["title"].tolist()

    selected_movie = st.selectbox(
        "Select a Movie:",
        movie_list
    )

    # Generate recommendation button
    if st.button("Get Recommendations"):

        results = recommend(
            selected_movie,
            df,
            tfidf_matrix,
            top_n=5
        )

        st.subheader("Recommended Movies")

        for i, movie in enumerate(results, start=1):

            st.write(
                f"{i}. **{movie['Movie']}**"
            )

            st.write(
                f"Similarity Score: "
                f"{movie['Similarity Score']}"
            )

            st.divider()

except Exception as e:

    st.error(f"Error: {e}")
