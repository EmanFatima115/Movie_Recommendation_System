# Movie Recommendation System

This project was completed as part of my **Machine Learning Internship at Syntecxhub**.

## Project Overview

The Movie Recommendation System recommends movies similar to a movie selected by the user.

For this project, I used a **content-based filtering** approach. Movie titles and genres were used as text features. These features were converted into TF-IDF vectors, and cosine similarity was used to find similar movies.

## Dataset

The dataset used in this project is the **MovieLens Latest-Small dataset**.

It contains:

- Movie ID
- Movie title
- Genres

## Data Cleaning and EDA

The following steps were performed:

- Loaded the MovieLens dataset using Pandas
- Checked dataset shape and columns
- Checked missing values
- Removed rows with missing titles or genres
- Checked duplicate records
- Cleaned movie genres
- Cleaned movie titles
- Created a combined content feature using movie titles and genres

## Recommendation Method

The system uses:

1. TF-IDF Vectorization
2. Cosine Similarity
3. Content-based filtering

The movie title and genres are combined into one text feature. TF-IDF converts this text into numerical vectors. Cosine similarity then measures the similarity between movies.

## Example

For **Lion King, The**, the system recommended:

- Lion King 1½, The
- Lion King II: Simba's Pride, The
- Wind and the Lion, The
- King and I, The
- Polar Express, The

## Qualitative Evaluation

The recommendation system was tested using sample movie queries such as:

- Jumanji
- Toy Story
- Lion King, The

The results were manually checked to evaluate whether the recommended movies were reasonably similar to the selected movie.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Cosine Similarity
- MovieLens Dataset

## Project Files

- `main.py` — Movie recommendation system implementation
- `movies.csv` — MovieLens dataset
- `README.md` — Project documentation

## How to Run

Install the required libraries:

```bash
pip install pandas scikit-learn
```

Run the project:

```bash
python main.py
```

## Internship

**Syntecxhub — Machine Learning Internship**

**Task 4 — Project 1: Movie Recommendation System**

**Author:** Eman Fatima

