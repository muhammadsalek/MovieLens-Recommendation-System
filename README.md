👉 Download MovieLens 100K Dataset (ml-100k.zip)

This dataset contains 100,000 ratings from 943 users on 1,682 movies. It's widely used for building and evaluating recommender systems.

📄 Final README for MovieLens 100K Recommendation System
Project Overview

This project implements a Top-N movie recommendation system using the MovieLens 100K dataset. The system employs collaborative filtering, matrix factorization, and evaluates models using standard metrics like Precision, Recall, and NDCG.

🧠 Approaches Implemented
a) Collaborative Filtering

User-Based CF (UBCF): Recommends movies based on similar users' preferences.

Similarity Measure: Cosine similarity.

Number of Neighbors: 30.

Item-Based CF (IBCF): Recommends movies similar to those a user liked.

Similarity Measure: Cosine similarity.

Number of Neighbors: 30.

b) Matrix Factorization

SVD (Singular Value Decomposition): Decomposes the user-item matrix into latent factors.

Latent Dimensions: 20.

Iterations: 20.

Trained using the recosystem package.

c) Optional Enhancement

Neural Embeddings / Deep Learning: Prepared for future experimentation using Keras.

📊 Data Preparation

Removed users with fewer than 5 ratings and movies with fewer than 10 ratings to reduce sparsity.

Created a realRatingMatrix for collaborative filtering.

Rating sparsity: ~90.98%.

Converted timestamps to datetime and analyzed rating trends over the years.

📈 Evaluation Metrics

Metrics computed on Top-10 recommendations:

Model	Precision	Recall	NDCG
SVD	0.0502	0.0075	0.0513
UBCF	0.0547	0.0094	0.0527
IBCF	0.0600	0.0112	0.0588

Observations:

IBCF slightly outperformed others in all metrics.

SVD captures latent factors but requires more tuning for better precision.

Overall precision and recall are low due to dataset sparsity; typical for sparse rating matrices.

🎬 Recommendation Function

A reusable R function recommend_movies(user_id, N, model_type) was implemented:

Input:

user_id: Integer

N: Number of movies to recommend

model_type: "SVD", "UBCF", "IBCF"

Output:

Top-N movie titles for the user.

Example Usage:

recommend_movies(user_id = 5, N = 10, model_type = "SVD")
recommend_movies(user_id = 5, N = 10, model_type = "UBCF")
recommend_movies(user_id = 5, N = 10, model_type = "IBCF")

🚀 Deployment

Saved all required objects for deployment (.rds files): movies, filtered ratings, rating matrix, models (SVD, UBCF, IBCF), Top-N SVD predictions.

Deployed the model to Hugging Face: MovieLens_App1

🔍 Findings & Recommendations

Best Approach: IBCF showed slightly better performance than UBCF and SVD in this dataset.

SVD Considerations: With hyperparameter tuning (dimensionality, iterations, learning rate), SVD could outperform CF methods.

Sparse Datasets: Precision/Recall is naturally low; filtering for frequent users and movies improves performance.

Future Enhancements:

Neural embeddings with Keras.

Hybrid model combining CF + content-based features.

Context-aware recommendations using timestamps, genres, or user demographics.

📂 Deliverables Summary
Deliverable	Status
Clean & documented R code	✅ Completed
Top-N recommendation function	✅ Implemented
Evaluation metrics (Precision, Recall, NDCG)	✅ Computed
Visualizations (ratings distribution, users, movies, metrics)	✅ Completed
Model deployment (Hugging Face)	✅ Completed

For more information and to access the dataset, visit the official GroupLens MovieLens 100K page:

👉 GroupLens MovieLens 100K Dataset
