🎬 Movie Recommendation System – MovieLens
Task Overview

The goal was to develop a Top-N movie recommendation system using the MovieLens dataset. The system should suggest movies for a given user, using collaborative filtering, matrix factorization, and optionally deep learning-based approaches, and evaluate them with standard metrics.

1️⃣ Approaches Implemented
a) Collaborative Filtering

User-Based CF (UBCF):

Similar users’ preferences are used to recommend unseen movies.

Similarity measure: Cosine similarity.

Number of neighbors: 30.

Item-Based CF (IBCF):

Movies similar to the ones a user liked are recommended.

Similarity measure: Cosine similarity.

Number of neighbors: 30.

b) Matrix Factorization

SVD (Singular Value Decomposition)

Reduced user-item matrix into latent factors.

Trained using recosystem with 20 latent dimensions and 20 iterations.

Can capture complex user-movie interactions.

c) Optional Enhancement

Neural embeddings / Deep learning (via keras) was prepared for future experimentation, though not fully implemented in this workflow.

2️⃣ Data Preparation

Removed users with <5 ratings and movies with <10 ratings to reduce sparsity.

Created a realRatingMatrix for collaborative filtering.

Rating sparsity: ~90.98% (typical for large-scale recommendation datasets).

Added timestamp → converted to datetime → analyzed rating trends over years.

3️⃣ Evaluation

Metrics computed on Top-10 recommendations:

Model	Precision	Recall	NDCG
SVD	0.0502	0.0075	0.0513
UBCF	0.0547	0.0094	0.0527
IBCF	0.0600	0.0112	0.0588

Observations:

IBCF slightly outperformed others in all metrics.

SVD captures latent factors but requires more tuning for better precision.

Overall precision and recall are low due to dataset sparsity; typical for sparse rating matrices.

Comparison Plot:

Saved as model_comparison.png to visualize Precision, Recall, and NDCG across models.

4️⃣ Recommendation Function

A reusable R function recommend_movies(user_id, N, model_type) was implemented:

Input:

user_id: integer

N: number of movies to recommend

model_type: "SVD", "UBCF", "IBCF"

Output:

Top-N movie titles for the user.

Example Usage:

recommend_movies(user_id = 5, N = 10, model_type = "SVD")
recommend_movies(user_id = 5, N = 10, model_type = "UBCF")
recommend_movies(user_id = 5, N = 10, model_type = "IBCF")

5️⃣ Deployment

Saved all required objects for deployment (.rds files): movies, filtered ratings, rating matrix, models (SVD, UBCF, IBCF), Top-N SVD predictions.

Deployed the model to Hugging Face:
MovieLens_App1

6️⃣ Findings / Recommendations

Best Approach: IBCF showed slightly better performance than UBCF and SVD in this dataset.

SVD Considerations: With hyperparameter tuning (dimensionality, iterations, learning rate), SVD could outperform CF methods.

Sparse Datasets: Precision/Recall is naturally low; filtering for frequent users and movies improves performance.

Future Enhancements:

Neural embeddings with Keras.

Hybrid model combining CF + content-based features.

Context-aware recommendations using timestamps, genres, or user demographics.

7️⃣ Deliverables Summary
Deliverable	Status
Clean & documented R code	✅ Completed
Top-N recommendation function	✅ Implemented
Evaluation metrics (Precision, Recall, NDCG)	✅ Computed
Visualizations (ratings distribution, users, movies, metrics)	✅ Completed
Model deployment (Hugging Face)	✅ Completed
