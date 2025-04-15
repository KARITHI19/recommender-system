# streamlit_app.py

import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import numpy as np

# Load the cleaned dataset
@st.cache_data
def load_data():
    reviews = pd.read_csv("Reviews.csv")
    reviews = reviews.dropna()
    reviews = reviews[['UserId', 'ProductId', 'Score']]
    return reviews

# Create sparse matrix
def create_matrix(reviews):
    user_map = {user: idx for idx, user in enumerate(reviews['UserId'].unique())}
    item_map = {item: idx for idx, item in enumerate(reviews['ProductId'].unique())}
    user_inv_map = {idx: user for user, idx in user_map.items()}
    item_inv_map = {idx: item for item, idx in item_map.items()}

    user_ids = reviews['UserId'].map(user_map)
    item_ids = reviews['ProductId'].map(item_map)

    data_matrix = csr_matrix((reviews['Score'], (user_ids, item_ids)))
    return data_matrix, user_map, item_map, user_inv_map, item_inv_map

# Train ALS model
def train_model(matrix):
    model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
    matrix = matrix.T.tocsr()
    model.fit(matrix)
    return model

# Recommend products for a user
def recommend(user_id, model, matrix, user_map, item_inv_map):
    if user_id not in user_map:
        return ["User ID not found."]
    
    user_idx = user_map[user_id]
    recommended = model.recommend(user_idx, matrix.T, N=5)
    return [item_inv_map[iid] for iid, score in recommended]

# Streamlit GUI
st.title("🛒 Product Recommender System")
reviews = load_data()

# Optional: Visualize score distribution
st.subheader("Rating Score Distribution")
score_counts = reviews['Score'].value_counts().sort_index()
st.bar_chart(score_counts)

# Train model
matrix, user_map, item_map, user_inv_map, item_inv_map = create_matrix(reviews)
model = train_model(matrix)

# User input
user_input = st.text_input("🔍 Enter a User ID to get product recommendations:")

if user_input:
    with st.spinner("Generating recommendations..."):
        results = recommend(user_input, model, matrix, user_map, item_inv_map)
        if isinstance(results, list):
            st.success("Here are your top recommendations:")
            for i, r in enumerate(results, 1):
                st.write(f"{i}. {r}")
        else:
            st.error(results)

