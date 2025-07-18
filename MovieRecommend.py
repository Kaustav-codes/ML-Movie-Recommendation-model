# Step 1: Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Step 2: Loading the .csv file
# title, genres, keywords, overview, production_companies, tagline are included in the .csv file
df = pd.read_csv('tmdb_5000_movies.csv')

# Step 3: Handle missing values in the relevant columns
# List of metadata columns to use
features = ['genres', 'keywords', 'overview', 'production_companies', 'tagline']
for feature in features:
    if feature in df.columns:
        # Replace NaN with empty string for existing column
        df[feature] = df[feature].fillna('')
    else:
        # If column is missing, create it with empty strings
        df[feature] = ''
        print(f"Column '{feature}' not found in data; created empty column for it.")

# Ensure the 'title' column exists and fill missing titles
if 'title' in df.columns:
    df['title'] = df['title'].fillna('')
else:
    raise KeyError("The dataset must have a 'title' column with movie titles.")

# Step 4: Createing a 'soup' by concatenating the text of all features
df['soup'] = (df['genres'] + ' ' + df['keywords'] + ' ' +
              df['overview'] + ' ' + df['tagline'] + ' ' +
              df['production_companies'])
print("Created 'soup' feature by combining genres, keywords, overview, tagline, and production companies.")

# Step 5: Compute TF-IDF matrix for the movie soups
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['soup'])
print(f"TF-IDF matrix has shape {tfidf_matrix.shape} (movies x features).")

# Step 6: Computing cosine similarity between movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print("Calculated cosine similarity matrix for all movies.")

# Step 7: Building a reverse map of movie titles to index
df['title_lower'] = df['title'].str.lower()
indices = pd.Series(df.index, index=df['title_lower'])

def get_recommendations(title, num_recommendations=5):
    """
    Given a movie title and number, return top similar movie titles.
    If title is not found, return an empty list.
    """
    title = title.lower()
    if title not in indices:
        print(f"Error: Movie title '{title}' not found. Please check the spelling and try again.")
        return []
    # Index of the movie that matches the title
    idx = indices[title]
    # Get pairwise similarity scores for this movie against all others
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies by similarity score (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Skip the first one since it is the movie itself (highest similarity)
    sim_scores = sim_scores[1:num_recommendations+1]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top similar movie titles
    return df['title'].iloc[movie_indices].tolist()

# Step 8: User input for movie title and number of recommendations
movie_title = input("Enter a movie title: ").strip()
try:
    num_recs = int(input("Enter number of recommendations to retrieve: "))
except ValueError:
    print("Invalid number. Using default of 5 recommendations.")
    num_recs = 5

# Step 9: Get and display recommendations
recommendations = get_recommendations(movie_title, num_recs)
if recommendations:
    print(f"\nTop {num_recs} recommendations similar to '{movie_title}':")
    for i, rec in enumerate(recommendations, start=1):
        print(f"{i}. {rec}")
