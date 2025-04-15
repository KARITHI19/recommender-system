# streamlit_app.py

import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import numpy as np

st.title("🛒 Product Recommender System")

# File upload
uploaded_file = st.file_uploader("📁 Upload your Reviews CSV file", type=["csv"])

if uploaded_file is not None:
    reviews = pd.read_csv(uploaded_file)

    # Basic validation
    required_cols = {'UserId', 'ProductId', 'Score'}
    if not required_cols.issubset(reviews.columns):
        st.error("Uploaded file must contain the following columns: UserId, ProductId, Score")
    else:
        reviews = reviews.dropna(subset=['UserId', 'ProductId', 'Score'])

        # Show score distribution
        st.subheader("📊 Score Distribution")
        score_counts = reviews['Score'].value_counts().sort_index()
        st.bar_chart(score_counts)

        # Create user-item matrix
        user_map = {user: idx for idx, user in enumerate(reviews['UserId'].unique())}
        item_map = {item: idx for idx, item in enumerate(reviews['ProductId'].unique())}
        user_inv_map = {idx: user for user, idx in user_map.items()}
        item_inv_map = {idx: item for item, idx in item_map.items()}

        user_ids = reviews['UserId'].map(user_map)
        item_ids = reviews['ProductId'].map(item_map)

        matrix = csr_matrix((reviews['Score'], (user_ids, item_ids)))

        # Train model
        with st.spinner("Training recommendation model..."):
            model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
            model.fit(matrix.T.tocsr())  # IMPORTANT: use CSR format after transpose
            st.success("Model trained successfully!")

        # Input for recommendations
        user_input = st.text_input("🔍 Enter a User ID to get recommendations:")

        if user_input:
            if user_input not in user_map:
                st.warning("User ID not found in the dataset.")
            else:
                user_idx = user_map[user_input]
                recommendations = model.recommend(user_idx, matrix.tocsr(), N=5)  # <-- FIXED HERE
                st.subheader("✅ Top Recommendations:")
                for i, (item_id, score) in enumerate(recommendations, 1):
                    st.write(f"{i}. Product ID: `{item_inv_map[item_id]}` (Score: {score:.2f})")
else:
    st.info("Please upload a CSV file to begin.")
