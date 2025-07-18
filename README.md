# 🎬 Movie Recommendation System using TF-IDF & Cosine Similarity

This project is a **Content-Based Movie Recommendation System** built using the TMDB 5000 Movies Dataset. It leverages **TF-IDF Vectorization** and **Cosine Similarity** to recommend movies based on a user-provided title by analyzing textual metadata like genres, keywords, overview, production companies, and tagline.

## 📂 Dataset

The project uses the [`tmdb_5000_movies.csv`](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) dataset, which includes features such as:
- `title`
- `genres`
- `keywords`
- `overview`
- `production_companies`
- `tagline`

## ScreenShots
<img width="2368" height="3986" alt="ML-Minor-code" src="https://github.com/user-attachments/assets/2ec87160-6cf6-418b-969b-23abb74acd62" />
<img width="1788" height="685" alt="ML-minor-Output" src="https://github.com/user-attachments/assets/b71c67f7-62c0-4c24-8983-97abe78b430a" />



## 🔍 Features Used for Recommendation

We combine the following metadata columns to form a textual "soup":
- `genres`
- `keywords`
- `overview`
- `tagline`
- `production_companies`

This "soup" is vectorized using **TfidfVectorizer** to create a sparse matrix for computing similarity.

## 🛠️ How It Works

1. Preprocess the dataset and fill missing values.
2. Concatenate relevant metadata columns into a single text feature (`soup`).
3. Use **TF-IDF** to vectorize the text and remove stop words.
4. Compute pairwise **cosine similarity** between all movie vectors.
5. Return the top N most similar movies for a given title.

## 📦 Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`

## ▶️ Running the Code

Make sure the dataset file `tmdb_5000_movies.csv` is in the same directory as the script.

```bash
python movie_recommender.py

