import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import re
# pip install googletrans==4.0.0-rc1
from googletrans import Translator

# Dataset
data = pd.read_csv('../avaliacao2/goodreads_data.csv')
data = data.drop('URL', axis=1)
data['Description'].fillna('', inplace=True)
data['Num_Ratings'] = data['Num_Ratings'].str.replace(',', '').astype(float)

# Dados para front-end
autores = data['Author'].unique()
data['Genres'] = data['Genres'].apply(lambda x: re.sub(r'[^\w\s,]', '', x))

generos_unicos = set()

for lista_generos in data['Genres'].str.split(', '):
    generos_unicos.update(lista_generos)

generos = list(generos_unicos)

min_avg_rating = data['Avg_Rating'].min()
max_avg_rating = data['Avg_Rating'].max()
min_num_ratings = data['Num_Ratings'].min()
max_num_ratings = data['Num_Ratings'].max()

# Tfid
tfidf_author = TfidfVectorizer(stop_words='english')
tfidf_description = TfidfVectorizer(stop_words='english')
tfidf_genres = TfidfVectorizer(stop_words='english')

Author_TFIDF = tfidf_author.fit_transform(data['Author'])
Description_TFIDF = tfidf_description.fit_transform(data['Description'])
Genres_TFIDF = tfidf_genres.fit_transform(data['Genres'])

# Normalização
scaler_avg_rating = MinMaxScaler(copy=True)
scaler_num_ratings = MinMaxScaler(copy=True)

Avg_Rating_Normalized = scaler_avg_rating.fit_transform(data[['Avg_Rating']])
Num_Ratings_Normalized = scaler_num_ratings.fit_transform(data[['Num_Ratings']])

data[['Avg_Rating_Normalized']] = Avg_Rating_Normalized
data[['Num_Ratings_Normalized']] = Num_Ratings_Normalized

# Função de tradução
def traduzir_texto(texto):
    translator = Translator()
    traducao = translator.translate(texto, src='pt', dest='en')
    return traducao.text

# Função auxiliar
def recommended_books(author_input, description_input, genres_input, avg_rating_input, num_ratings_input):
    data_copy = data.copy()

    author_input_tfidf = tfidf_author.transform([author_input])
    author_cosine_similarities = cosine_similarity(author_input_tfidf, Author_TFIDF).flatten()

    if description_input:
        description_input = traduzir_texto(description_input)

    description_input_tfidf = tfidf_description.transform([description_input])
    description_cosine_similarities = cosine_similarity(description_input_tfidf, Description_TFIDF).flatten()

    genres_input_str = ', '.join(genres_input)

    genres_input_tfidf = tfidf_genres.transform([genres_input_str])
    genres_cosine_similarities = cosine_similarity(genres_input_tfidf, Genres_TFIDF).flatten()

    data_copy['Author_Rank'] = author_cosine_similarities
    data_copy['Author_Rank'] = data_copy['Author_Rank'].rank(ascending=False)

    data_copy['Description_Rank'] = description_cosine_similarities
    data_copy['Description_Rank'] = data_copy['Description_Rank'].rank(ascending=False)

    data_copy['Genres_Rank'] = genres_cosine_similarities
    data_copy['Genres_Rank'] = data_copy['Genres_Rank'].rank(ascending=False)

    avg_rating_input_normalized = scaler_avg_rating.transform([[avg_rating_input]])[0][0]
    num_ratings_input_normalized = scaler_num_ratings.transform([[num_ratings_input]])[0][0]

    rating_values = np.array([[avg_rating_input_normalized, num_ratings_input_normalized]])

    data_copy['Ratings_Rank'] = euclidean_distances(rating_values, data_copy[['Avg_Rating_Normalized', 'Num_Ratings_Normalized']])[0]
    data_copy['Ratings_Rank'] = data_copy['Ratings_Rank'].rank(ascending=True)

    data_copy['Overall_Rank'] = data_copy[['Author_Rank', 'Description_Rank', 'Genres_Rank', 'Ratings_Rank']].mean(axis=1)

    top_df = data_copy.sort_values(by='Overall_Rank').head(10)

    top_df = top_df[['Book','Author', 'Description', 'Genres', 'Avg_Rating', 'Num_Ratings']]

    return top_df
