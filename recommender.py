import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import numpy as np

st.title("🛒 Product Recommender System")

# Google Drive shareable file ID
file_id = "1lmr0-xs2HBhsyj14kerFCKTrsNnVzQv7"  # replace with your actual file ID
url = f"https://drive.google.com/drive/u/0/home={file_id}&export=download"

# Load CSV directly from Google Drive
@st.cache_data
def load_data_from_drive(url):
    return pd.read_csv(url)

with st.spinner("Loading dataset from Google Drive..."):
    try:
        reviews = load_data_from_drive(url)
        st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

# Validate the necessary columns
required_cols = {'UserId', 'ProductId', 'Score'}
if not required_cols.issubset(reviews.columns):
    st.error("CSV must contain: UserId, ProductId, Score")
else:
    reviews = reviews.dropna(subset=['UserId', 'ProductId', 'Score'])

    st.subheader("📊 Score Distribution")
    st.bar_chart(reviews['Score'].value_counts().sort_index())

    # Map users and products to indices
    user_map = {user: idx for idx, user in enumerate(reviews['UserId'].unique())}
    item_map = {item: idx for idx, item in enumerate(reviews['ProductId'].unique())}
    user_inv_map = {idx: user for user, idx in user_map.items()}
    item_inv_map = {idx: item for item, idx in item_map.items()}

    user_ids = reviews['UserId'].map(user_map)
    item_ids = reviews['ProductId'].map(item_map)
    matrix = csr_matrix((reviews['Score'], (user_ids, item_ids)))

    # Train the ALS model
    with st.spinner("Training recommendation model..."):
        model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
        model.fit(matrix.T.tocsr())
        st.success("Model trained!")

    # Input for user recommendation
    user_input = st.text_input("🔍 Enter a User ID to get recommendations:")

    if user_input:
        try:
            user_input = int(user_input)
            if user_input not in user_map:
                st.warning("User ID not found.")
            else:
                user_idx = user_map[user_input]
                recommendations = model.recommend(user_idx, matrix.tocsr(), N=5)

                st.subheader("✅ Top Recommendations:")
                for i, (item_id, score) in enumerate(recommendations, 1):
                    st.write(f"{i}. Product ID: `{item_inv_map[item_id]}` (Score: {score:.2f})")
        except ValueError:
            st.warning("Please enter a valid integer for User ID.")
