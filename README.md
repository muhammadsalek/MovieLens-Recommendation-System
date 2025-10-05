MovieLens Recommender System
📌 Project Overview

This project implements a movie recommendation system using the MovieLens dataset. Users receive personalized movie suggestions based on their previous ratings. The system applies:

Collaborative Filtering – Recommendations based on user ratings.

Content-Based Filtering – Recommendations based on movie features (e.g., genre, director).

Hybrid Approach – Combines collaborative and content-based methods.

🧪 Tools and Libraries Used

R / RStudio – Primary programming environment

dplyr / data.table – Data manipulation

recommenderlab / recosystem – Recommender system implementation

ggplot2 / plotly – Data visualization

shiny / shinyWidgets / shinycssloaders – Interactive web app

📁 Dataset

The dataset used is MovieLens 1M, containing over 1,000,000 ratings. It is available at GroupLens
.

🚀 Project Structure
/MovieLens_Recommender
│
├── data/                 # Dataset files
├── R/                    # R scripts
│   ├── data_preprocessing.R
│   ├── model_building.R
│   └── evaluation.R
├── app/                  # Shiny app files
│   ├── ui.R
│   └── server.R
└── README.md             # This file

🛠️ How to Run

Install required R packages:

install.packages(c("dplyr","data.table","recommenderlab","recosystem","ggplot2","plotly","shiny","shinyWidgets","shinycssloaders"))


Load and preprocess the data:

source("R/data_preprocessing.R")
processed_data <- preprocess_data("data/movielens-1m.csv")


Build and train the recommendation model:

source("R/model_building.R")
model <- train_model(processed_data)


Evaluate the model:

source("R/evaluation.R")
evaluate_model(model, processed_data)


Launch the Shiny app:

shiny::runApp("app/")

📄 License

This project is licensed under the MIT License. See the LICENSE
 file for details.

📧 Contact

Email: salekml@example.com

GitHub: https://github.com/salekml

LinkedIn: https://www.linkedin.com/in/salekml
