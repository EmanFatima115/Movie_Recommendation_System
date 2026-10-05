import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# EDA:
movies=pd.read_csv('movies.csv')
print(movies.shape)
print(movies.head())
print(movies.columns)
print(movies.isnull().sum())
# Data Cleaning:
movies=movies.dropna(subset=['title','genres'])
print(movies.shape)
print(movies.duplicated().sum())
movies['genres']=movies['genres'].str.replace('|',' ',regex=False)
print(movies.head())
movies["title"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
print(movies[['title','genres']].head())
movies['content']=movies['title'] +' '+ movies['genres']
print(movies[['title','genres','content']].head())
# Text Features:
tfidf=TfidfVectorizer()
tfidf_matrix=tfidf.fit_transform(movies['content'])
print(tfidf_matrix.shape)
similarity=cosine_similarity(tfidf_matrix)
print(similarity.shape)
print(similarity[0][:10])
# Recommendation And Cosine similarity:
def recommend_movies(movies_title):


    movie_index=movies[movies['title']==movies_title].index[0]
    print('movie index', movie_index)
    similarity_scores = list(enumerate(similarity[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    for index, score in similarity_scores[1:6]:
        print(movies.iloc[index]['title'], score)


recommend_movies('Jumanji')
recommend_movies('Toy Story')
recommend_movies('Lion King, The')
# Evaluation:
# Qualitative Evaluation
# The recommendations were checked using sample movie queries.
# The system produced relevant movies for similar titles and genres.
