import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import streamlit as st
import re
from dataset import *

def recommend_app():
    st.title("Sistema de Recomendação de Livros")

    autor_input = st.selectbox("Selecione o Autor", autores)
    
    description_input = st.text_area("Descrição")
    
    generos_selecionados = st.multiselect("Selecione os Gêneros", generos)

    avg_rating_input = st.slider("Selecione o Rating Médio", min_avg_rating, max_avg_rating, (min_avg_rating + max_avg_rating) / 2)
    num_ratings_input = st.slider("Selecione o Número de Ratings", min_num_ratings, max_num_ratings, (min_num_ratings + max_num_ratings) / 2)

    if st.button("Obter Recomendações"):
        # Transformar novas entradas com os modelos TF-IDF existentes
        author_new = tfidf_author.transform([autor_input])

        # Calcular a similaridade de cosseno entre a nova entrada do autor e os autores existentes
        cosine_similarities = cosine_similarity(author_new, Author_TFIDF)

        # Ordenar os índices em ordem decrescente de similaridade
        similar_books_indices = cosine_similarities.argsort()[0][::-1]

        # Recomendar os 10 livros mais similares
        recommended_books = data.iloc[similar_books_indices[:10]]

        # Exibir os livros recomendados
        st.write("Aqui estão suas recomendações:")
        st.write(recommended_books)

recommend_app()

