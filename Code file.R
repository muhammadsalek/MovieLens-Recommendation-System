









# CRAN packages
install.packages("shiny")
install.packages("dplyr")
install.packages("shinycssloaders")
install.packages("shinyWidgets")
install.packages("recommenderlab")

# For recosystem (from CRAN)
install.packages("recosystem")

# Optional: dependencies for shinyWidgets and shinycssloaders
install.packages("htmltools")
install.packages("RColorBrewer")
install.packages("magrittr")
install.packages("data.table")









#### 🎬 MovieLens Recommendation System ####
#### Required Libraries ####


# 1️⃣ Data Handling
install.packages("data.table")   # দ্রুত ডেটা লোড ও প্রক্রিয়াকরণ
# 2️⃣ Recommendation Algorithms
install.packages("recommenderlab")   # UBCF, IBCF, evaluation tools

# 3️⃣ Visualization (optional but helpful)
install.packages("ggplot2")      # Evaluation plotting (Precision/Recall curves)
install.packages("gridExtra")    # একাধিক প্লট একসাথে দেখাতে

install.packages("caret")        # Machine learning helper tools (optional)

# 5️⃣ Neural Embedding (optional advanced)
install.packages("keras")        # Deep learning-based recommendation (optional)

# 6️⃣ Deployment API (for Hugging Face)
install.packages("plumber")      # API তৈরি করার জন্য

# 7️⃣ Miscellaneous
install.packages("stringr")      # string manipulation
install.packages("Matrix")       # sparse matrix operations
install.packages("readr")        # fast CSV/TSV reading







# 1. Install / Load Packages (non-interactive)
# Non-interactive install helper (only install if missing)
packages <- c(
  "data.table","dplyr","tidyr","reshape2",
  "recommenderlab","recosystem","ggplot2","gridExtra",
  "Metrics","caret","keras","plumber","stringr","Matrix","readr"
)


install_if_missing <- function(pkgs){
  for(p in pkgs){
    if(!suppressWarnings(requireNamespace(p, quietly = TRUE))){
      message("Installing: ", p)
      if(p == "recommenderlab"){
        # try CRAN first, then BiocManager
        try(install.packages("recommenderlab", dependencies=TRUE), silent = TRUE)
        if(!requireNamespace("recommenderlab", quietly = TRUE)){
          if(!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
          BiocManager::install("recommenderlab", ask = FALSE)
        }
      } else {
        install.packages(p, dependencies = TRUE)
      }
    }
  }
}


install_if_missing(packages)


#### Load Required Packages ####
library(data.table)
library(dplyr)
library(tidyr)
library(reshape2)
library(recommenderlab)
library(recosystem)
library(ggplot2)
library(gridExtra)
library(Metrics)
library(caret)
library(keras)
library(plumber)
library(stringr)
library(Matrix)
library(readr)












#### ===============================
#### MovieLens Recommendation System - Full Workflow
#### ===============================

# Load libraries
library(data.table)
library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)
library(lubridate)
library(recommenderlab)
library(recosystem)

# 1️⃣ Set base path and load datasets
base_path <- "D:/job task/data/ml-100k/ml-100k/"

ratings <- fread(file.path(base_path, "u.data"), sep="\t",
                 col.names=c("userId","movieId","rating","timestamp"))

movies <- fread(file.path(base_path, "u.item"), sep="|", header=FALSE, quote="",
                encoding="Latin-1", fill=TRUE) %>% select(movieId=V1, title=V2)

#### ===============================
#### 2️⃣ Explore Ratings
#### ===============================
head(ratings)
str(ratings)
summary(ratings)
colSums(is.na(ratings))
length(unique(ratings$userId))
length(unique(ratings$movieId))
table(ratings$rating)

ggplot(ratings, aes(x=rating)) +
  geom_bar(fill="steelblue") +
  labs(title="Rating Distribution", x="Rating", y="Count")

# Ratings per user
user_ratings <- ratings %>% group_by(userId) %>% summarise(n_ratings = n())
summary(user_ratings$n_ratings)
ggplot(user_ratings, aes(x=n_ratings)) + 
  geom_histogram(bins=30, fill="orange") +
  scale_x_log10() +
  labs(title="Number of Ratings per User", x="Number of Ratings (log scale)", y="Count of Users")

# Ratings per movie
movie_ratings <- ratings %>% group_by(movieId) %>% summarise(n_ratings = n())
summary(movie_ratings$n_ratings)
ggplot(movie_ratings, aes(x=n_ratings)) + 
  geom_histogram(bins=30, fill="purple") +
  scale_x_log10() +
  labs(title="Number of Ratings per Movie", x="Number of Ratings (log scale)", y="Count of Movies")

# Average movie ratings
avg_movie_ratings <- ratings %>%
  group_by(movieId) %>%
  summarise(avg_rating = mean(rating), n_ratings = n()) %>%
  inner_join(movies, by="movieId")

top_movies <- avg_movie_ratings %>% filter(n_ratings >= 50) %>%
  arrange(desc(avg_rating)) %>% head(10)
top_movies

#### ===============================
#### 3️⃣ Filter sparse data
#### ===============================
ratings_filtered <- ratings %>%
  group_by(userId) %>% filter(n() >= 5) %>%
  ungroup() %>%
  group_by(movieId) %>% filter(n() >= 10) %>%
  ungroup()

num_users <- length(unique(ratings_filtered$userId))
num_movies <- length(unique(ratings_filtered$movieId))
num_ratings <- nrow(ratings_filtered)
sparsity <- 1 - (num_ratings / (num_users * num_movies))
sparsity

# Add datetime column
ratings_filtered <- ratings_filtered %>%
  mutate(rating_date = as_datetime(timestamp))

ggplot(ratings_filtered, aes(x=year(rating_date))) +
  geom_bar(fill="skyblue") +
  labs(title="Ratings Over Years", x="Year", y="Count")

#### ===============================
#### 4️⃣ Create Rating Matrix
#### ===============================
rating_matrix <- ratings_filtered %>%
  select(userId, movieId, rating) %>%
  pivot_wider(names_from = movieId, values_from = rating) %>%
  column_to_rownames("userId") %>%
  as.matrix() %>%
  as("realRatingMatrix")

#### ===============================
#### 5️⃣ Train/Test Split
#### ===============================
set.seed(123)
train_index <- sample(x = c(TRUE, FALSE), size = nrow(rating_matrix), replace = TRUE, prob = c(0.8, 0.2))
train <- rating_matrix[train_index, ]
test  <- rating_matrix[!train_index, ]

#### ===============================
#### 6️⃣ Build Collaborative Filtering Models
#### ===============================
# User-Based CF
ubcf_model <- Recommender(train, method = "UBCF", param = list(method = "Cosine", nn = 30))
pred_ubcf <- predict(ubcf_model, test, type = "topNList", n = 10)
pred_list_ubcf <- as(pred_ubcf, "list")
recommended_titles_ubcf <- lapply(pred_list_ubcf, function(ids) {
  movies$title[movies$movieId %in% as.numeric(ids)]
})

# Item-Based CF
ibcf_model <- Recommender(train, method = "IBCF", param = list(method = "Cosine", k = 30))
pred_ibcf <- predict(ibcf_model, test, type = "topNList", n = 10)
pred_list_ibcf <- as(pred_ibcf, "list")
recommended_titles_ibcf <- lapply(pred_list_ibcf, function(ids) {
  movies$title[movies$movieId %in% as.numeric(ids)]
})

#### ===============================
#### 7️⃣ SVD / Matrix Factorization
#### ===============================
train_file <- tempfile()
write.table(ratings_filtered %>% select(userId, movieId, rating), 
            file = train_file, sep = " ", row.names = FALSE, col.names = FALSE)

r <- Reco()
r$train(data_file(train_file), opts = list(dim = 20, lrate = 0.1, niter = 20, verbose = TRUE))

# Example: Predict Top-N for a user
recommend_movies <- function(user_id, N = 10, model_type = "SVD") {
  
  if(model_type == "SVD") {
    user_movies <- ratings_filtered %>% filter(userId == user_id) %>% pull(movieId)
    unseen_movies <- setdiff(unique(ratings_filtered$movieId), user_movies)
    unseen_df <- data.frame(user = user_id, item = unseen_movies)
    pred_ratings <- r$predict(data_memory(user = unseen_df$user, item = unseen_df$item))
    
    top_n_df <- data.frame(movieId = unseen_movies, pred_rating = pred_ratings) %>%
      arrange(desc(pred_rating)) %>% head(N)
    
    movies$title[movies$movieId %in% top_n_df$movieId]
    
  } else if(model_type == "UBCF") {
    pred <- predict(ubcf_model, rating_matrix[user_id,], type="topNList", n=N)
    ids <- as(pred, "list")[[1]]
    movies$title[movies$movieId %in% as.numeric(ids)]
    
  } else if(model_type == "IBCF") {
    pred <- predict(ibcf_model, rating_matrix[user_id,], type="topNList", n=N)
    ids <- as(pred, "list")[[1]]
    movies$title[movies$movieId %in% as.numeric(ids)]
    
  } else stop("Invalid model_type. Choose 'SVD', 'UBCF', or 'IBCF'.")
}

# Example usage
recommend_movies(user_id = 5, N = 10, model_type = "SVD")
recommend_movies(user_id = 5, N = 10, model_type = "UBCF")
recommend_movies(user_id = 5, N = 10, model_type = "IBCF")

#### ===============================
#### 8️⃣ Evaluate Models (Precision@K, Recall@K, NDCG)
#### ===============================
compute_metrics <- function(topN_list, test_matrix, K = 10, goodRating = 4) {
  precisions <- c(); recalls <- c(); ndcgs <- c()
  
  for (i in 1:length(topN_list)) {
    user_id <- names(topN_list)[i]
    recommended <- topN_list[[i]]
    user_ratings <- as(test_matrix[user_id, ], "matrix")[1, ]
    actual <- which(user_ratings >= goodRating)
    
    p <- length(intersect(recommended[1:K], actual)) / K
    r <- if(length(actual) > 0) length(intersect(recommended[1:K], actual)) / length(actual) else 0
    rel <- ifelse(recommended[1:K] %in% actual, 1, 0)
    dcg <- sum(rel / log2(2:(K+1)))
    idcg <- sum(1 / log2(2:(min(K, length(actual)) + 1)))
    ndcg <- ifelse(idcg == 0, 0, dcg / idcg)
    
    precisions <- c(precisions, p)
    recalls <- c(recalls, r)
    ndcgs <- c(ndcgs, ndcg)
  }
  
  data.frame(
    Precision = mean(precisions, na.rm = TRUE),
    Recall = mean(recalls, na.rm = TRUE),
    NDCG = mean(ndcgs, na.rm = TRUE)
  )
}

# Compute metrics
topN_svd_list <- lapply(1:nrow(rating_matrix), function(u) {
  user_movies <- ratings_filtered %>% filter(userId == as.numeric(rownames(rating_matrix)[u])) %>% pull(movieId)
  unseen_movies <- setdiff(unique(ratings_filtered$movieId), user_movies)
  unseen_df <- data.frame(user = as.numeric(rownames(rating_matrix)[u]), item = unseen_movies)
  pred_ratings <- r$predict(data_memory(user = unseen_df$user, item = unseen_df$item))
  unseen_movies[order(-pred_ratings)][1:10]
})
names(topN_svd_list) <- rownames(rating_matrix)

topN_ubcf_list <- lapply(1:nrow(rating_matrix), function(u) {
  pred <- predict(ubcf_model, rating_matrix[u, ], type="topNList", n=10)
  as(pred, "list")[[1]]
})
names(topN_ubcf_list) <- rownames(rating_matrix)

topN_ibcf_list <- lapply(1:nrow(rating_matrix), function(u) {
  pred <- predict(ibcf_model, rating_matrix[u, ], type="topNList", n=10)
  as(pred, "list")[[1]]
})
names(topN_ibcf_list) <- rownames(rating_matrix)

metrics_svd <- compute_metrics(topN_svd_list, rating_matrix, K=10)
metrics_ubcf <- compute_metrics(topN_ubcf_list, rating_matrix, K=10)
metrics_ibcf <- compute_metrics(topN_ibcf_list, rating_matrix, K=10)
metrics_svd; metrics_ubcf; metrics_ibcf

# Comparison Plot
metrics <- data.frame(
  Model = c("SVD", "UBCF", "IBCF"),
  Precision = c(metrics_svd$Precision, metrics_ubcf$Precision, metrics_ibcf$Precision),
  Recall = c(metrics_svd$Recall, metrics_ubcf$Recall, metrics_ibcf$Recall),
  NDCG = c(metrics_svd$NDCG, metrics_ubcf$NDCG, metrics_ibcf$NDCG)
)
metrics_long <- tidyr::pivot_longer(metrics, cols = Precision:NDCG, names_to = "Metric", values_to = "Value")
ggplot(metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Comparison of Recommendation Models", y = "Metric Value")






####  Save Objects for Deployment ####

saveRDS(movies, "D:/job task/ML/MovieLensShinyApp/movies.rds")
saveRDS(ratings_filtered, "D:/job task/ML/MovieLensShinyApp/ratings_filtered.rds")
saveRDS(rating_matrix, "D:/job task/ML/MovieLensShinyApp/rating_matrix.rds")
saveRDS(r, "D:/job task/ML/MovieLensShinyApp/svd_model.rds")
saveRDS(ubcf_model, "D:/job task/ML/MovieLensShinyApp/ubcf_model.rds")
saveRDS(ibcf_model, "D:/job task/ML/MovieLensShinyApp/ibcf_model.rds")
saveRDS(topN_svd_list, "D:/job task/ML/MovieLensShinyApp/topN_svd.rds")



















#### ===============================
#### 10️⃣ Final Recommendation Function & Deployment Prep
#### ===============================

library(dplyr)
library(recommenderlab)
library(recosystem)
library(tibble)
library(ggplot2)
library(lubridate)

# Reusable function for Top-N recommendations
recommend_movies <- function(user_id, N = 10, model_type = "SVD") {
  
  if(model_type == "SVD") {
    # Movies already rated by user
    user_movies <- ratings_filtered %>% filter(userId == user_id) %>% pull(movieId)
    unseen_movies <- setdiff(unique(ratings_filtered$movieId), user_movies)
    
    unseen_df <- data.frame(user = user_id, item = unseen_movies)
    pred_ratings <- r$predict(data_memory(user = unseen_df$user, item = unseen_df$item))
    
    top_n_df <- data.frame(movieId = unseen_movies, pred_rating = pred_ratings) %>%
      arrange(desc(pred_rating)) %>%
      head(N)
    
    movies$title[movies$movieId %in% top_n_df$movieId]
    
  } else if(model_type == "UBCF") {
    pred <- predict(ubcf_model, rating_matrix[user_id,], type="topNList", n=N)
    ids <- as(pred, "list")[[1]]
    movies$title[movies$movieId %in% as.numeric(ids)]
    
  } else if(model_type == "IBCF") {
    pred <- predict(ibcf_model, rating_matrix[user_id,], type="topNList", n=N)
    ids <- as(pred, "list")[[1]]
    movies$title[movies$movieId %in% as.numeric(ids)]
    
  } else {
    stop("Invalid model_type. Choose 'SVD', 'UBCF', or 'IBCF'.")
  }
}

# Example usage
recommend_movies(user_id = 5, N = 10, model_type = "SVD")
recommend_movies(user_id = 5, N = 10, model_type = "UBCF")
recommend_movies(user_id = 5, N = 10, model_type = "IBCF")

#### ===============================
#### Save All Objects for Deployment (New Path)
#### ===============================
saveRDS(movies, "D:/job task/ML/MovieLensShinyApp/movies.rds")
saveRDS(ratings_filtered, "D:/job task/ML/MovieLensShinyApp/ratings_filtered.rds")
saveRDS(rating_matrix, "D:/job task/ML/MovieLensShinyApp/rating_matrix.rds")
saveRDS(r, "D:/job task/ML/MovieLensShinyApp/svd_model.rds")
saveRDS(ubcf_model, "D:/job task/ML/MovieLensShinyApp/ubcf_model.rds")
saveRDS(ibcf_model, "D:/job task/ML/MovieLensShinyApp/ibcf_model.rds")
saveRDS(topN_svd_list, "D:/job task/ML/MovieLensShinyApp/topN_svd.rds")

#### ===============================
#### Short Report Preparation
#### ===============================
# Compare metrics of SVD, UBCF, IBCF
metrics <- data.frame(
  Model = c("SVD", "UBCF", "IBCF"),
  Precision = c(metrics_svd$Precision, metrics_ubcf$Precision, metrics_ibcf$Precision),
  Recall = c(metrics_svd$Recall, metrics_ubcf$Recall, metrics_ibcf$Recall),
  NDCG = c(metrics_svd$NDCG, metrics_ubcf$NDCG, metrics_ibcf$NDCG)
)

metrics_long <- tidyr::pivot_longer(metrics, cols = Precision:NDCG, names_to = "Metric", values_to = "Value")

ggplot(metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Comparison of Recommendation Models", y = "Metric Value")

# Save metrics plot for report
ggsave("D:/job task/ML/MovieLensShinyApp/model_comparison.png", width = 8, height = 5)























































library(shiny)
library(dplyr)
library(shinycssloaders)
library(shinyWidgets)
library(recosystem)
library(recommenderlab)  # If necessary

# ==============================
# Load Data
# ==============================
movies <- readRDS("D:/job task/ML/MovieLensShinyApp/movies.rds")
ratings_filtered <- readRDS("D:/job task/ML/MovieLensShinyApp/ratings_filtered.rds")
rating_matrix <- readRDS("D:/job task/ML/MovieLensShinyApp/rating_matrix.rds")
r <- readRDS("D:/job task/ML/MovieLensShinyApp/svd_model.rds")
ubcf_model <- readRDS("D:/job task/ML/MovieLensShinyApp/ubcf_model.rds")
ibcf_model <- readRDS("D:/job task/ML/MovieLensShinyApp/ibcf_model.rds")
topN_svd_list <- readRDS("D:/job task/ML/MovieLensShinyApp/topN_svd.rds")

# Ensure poster_url exists
if(!"poster_url" %in% colnames(movies)){
  movies$poster_url <- "https://via.placeholder.com/150x225.png?text=No+Poster"
} else {
  movies$poster_url <- ifelse(is.na(movies$poster_url),
                              "https://via.placeholder.com/150x225.png?text=No+Poster",
                              movies$poster_url)
}

# ==============================
# Recommendation function
# ==============================
recommend_movies <- function(user_id, N = 10, model_type = "SVD") {
  
  if(!(user_id %in% rownames(rating_matrix))) {
    return(data.frame(title="No recommendations available", 
                      poster_url="https://via.placeholder.com/150x225.png?text=No+Poster"))
  }
  
  if(model_type == "SVD") {
    user_movies <- ratings_filtered %>% filter(userId == user_id) %>% pull(movieId)
    unseen_movies <- setdiff(unique(ratings_filtered$movieId), user_movies)
    
    if(length(unseen_movies) == 0) {
      return(data.frame(title="No recommendations available", 
                        poster_url="https://via.placeholder.com/150x225.png?text=No+Poster"))
    }
    
    unseen_df <- data.frame(user = user_id, item = unseen_movies)
    pred_ratings <- r$predict(data_memory(user = unseen_df$user, item = unseen_df$item))
    
    top_n_df <- data.frame(movieId = unseen_movies, pred_rating = pred_ratings) %>%
      arrange(desc(pred_rating)) %>%
      head(N)
    
    recs <- movies %>% filter(movieId %in% top_n_df$movieId) %>%
      select(title, poster_url)
    
  } else if(model_type == "UBCF") {
    pred <- predict(ubcf_model, rating_matrix[user_id,], type="topNList", n=N)
    ids <- as(pred, "list")[[1]]
    recs <- movies %>% filter(movieId %in% as.numeric(ids)) %>%
      select(title, poster_url)
    
  } else if(model_type == "IBCF") {
    pred <- predict(ibcf_model, rating_matrix[user_id,], type="topNList", n=N)
    ids <- as(pred, "list")[[1]]
    recs <- movies %>% filter(movieId %in% as.numeric(ids)) %>%
      select(title, poster_url)
    
  } else {
    stop("Invalid model_type. Choose 'SVD', 'UBCF', or 'IBCF'.")
  }
  
  if(nrow(recs) == 0) {
    recs <- data.frame(title="No recommendations available", 
                       poster_url="https://via.placeholder.com/150x225.png?text=No+Poster")
  }
  
  return(recs)
}

# ==============================
# Shiny UI
# ==============================
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      /* Header Gradient Animation */
      .header-emoji {
        font-size: 34px;
        font-weight: bold;
        background: linear-gradient(90deg, #ff4e50, #f9d423);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
      }
      @keyframes gradient {
        0% {background-position: 0%}
        50% {background-position: 100%}
        100% {background-position: 0%}
      }

      body { background-color: #fff0f5; }

      .btn-success { 
        background: linear-gradient(45deg, #ff6b6b, #fbc531);
        color: #fff;
        font-weight: bold;
        transition: transform 0.2s;
      }
      .btn-success:hover {
        transform: scale(1.05);
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
      }

      .poster img { transition: transform 0.3s; border-radius:10px; }
      .poster img:hover { transform: scale(1.1) rotate(-2deg); box-shadow: 0px 4px 20px rgba(0,0,0,0.5); }

      .title { text-align:center; font-size:12px; font-weight:bold; margin-top:5px; }

      .recommendation-scroll { max-height: 600px; overflow-y: auto; }

      .tags-overlay { position: relative; }
      .tags-overlay span {
        position: absolute;
        top: 5px; left: 5px;
        font-size: 16px;
        animation: sparkle 1.5s infinite;
      }
      @keyframes sparkle {
        0%, 100% {opacity: 1;}
        50% {opacity: 0.3;}
      }
    "))
  ),
  
  titlePanel(tags$div("🎥 MovieLens Recommendation System 🍿", class="header-emoji")),
  
  sidebarLayout(
    sidebarPanel(
      numericInput("user_id", "Enter User ID:", value = 1, min = 1, max = nrow(rating_matrix)),
      selectInput("model_type", "Select Model:", choices = c("SVD", "UBCF", "IBCF"), selected = "UBCF"),
      numericInput("top_n", "Number of Recommendations:", value = 10, min = 1, max = 20),
      actionBttn("recommend_btn", "GET RECOMMENDATIONS ✨", style = "material-flat", color = "success")
    ),
    
    mainPanel(
      h4("Recommended Movies 🎬:", style="color:darkblue;"),
      withSpinner(uiOutput("recommendations_ui"), type = 6)
    )
  )
)

# ==============================
# Shiny Server
# ==============================
server <- function(input, output) {
  
  recommendations <- eventReactive(input$recommend_btn, {
    recommend_movies(user_id = input$user_id, N = input$top_n, model_type = input$model_type)
  })
  
  output$recommendations_ui <- renderUI({
    recs <- recommendations()
    div(class="recommendation-scroll",
        fluidRow(
          lapply(1:nrow(recs), function(i){
            column(2,
                   div(class="poster tags-overlay",
                       tags$img(src = recs$poster_url[i], width="150px", height="225px"),
                       tags$span("✨"),
                       tags$p(recs$title[i], class="title")
                   )
            )
          })
        )
    )
  })
}

# ==============================
# Run App
# ==============================
shinyApp(ui = ui, server = server)





















































