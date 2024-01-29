import streamlit as st
from system import *
from style import page_style

def recommend_app():
    st.markdown(page_style, unsafe_allow_html=True)

    st.title("Sistema de Recomendação de Livros")

    author_input = st.selectbox("Qual autor escreve bons livros?", autores)
    
    description_input = st.text_area("Imagine uma descrição para um livro:")
    
    genres_input = st.multiselect("Selecione os gêneros que mais gosta:", generos)

    avg_rating_input = st.slider("Quão bem avaliado você espera que seja?", min_avg_rating, max_avg_rating, (min_avg_rating + max_avg_rating) / 2)

    num_ratings_input = st.slider("Escolha o número de avaliações que acredita ser relevante:", min_num_ratings, max_num_ratings, (min_num_ratings + max_num_ratings) / 2)

    if st.button("Obter Recomendações"):
        books = recommended_books(author_input, description_input, genres_input, avg_rating_input, num_ratings_input)
        st.write("Aqui estão suas recomendações:")
        for index, livro in books.iterrows():
            st.markdown(f"**Livro {index + 1}:** _{livro['Book']}_")
            st.markdown(f"Descrição: _{livro['Description']}_")
            st.markdown(f"Autor: _{livro['Author']}_")
            st.markdown(f"Gêneros: _{livro['Genres']}_")
            st.markdown(f"Avaliação Média: _{livro['Avg_Rating']}_")
            st.markdown(f"Número de Avaliações: _{livro['Num_Ratings']}_")
            st.markdown("---")

recommend_app()

