import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import re

data = pd.read_csv('../avaliacao2/goodreads_data.csv')

data['Description'].fillna('', inplace=True)

data['Num_Ratings'] = data['Num_Ratings'].str.replace(',', '').astype(float)

scaler_avg_rating = MinMaxScaler()
scaler_num_ratings = MinMaxScaler()

tfidf_author = TfidfVectorizer()
tfidf_description = TfidfVectorizer()
tfidf_genres = TfidfVectorizer()

Author_TFIDF = tfidf_author.fit_transform(data['Author'])
Description_TFIDF = tfidf_description.fit_transform(data['Description'])
Genres_TFIDF = tfidf_genres.fit_transform(data['Genres'])

Avg_Rating_Normalized = scaler_avg_rating.fit_transform(data[['Avg_Rating']])
Num_Ratings_Normalized = scaler_num_ratings.fit_transform(data[['Num_Ratings']])

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