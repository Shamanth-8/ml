# ml
machine learning projects are visible here
Data Science Notebooks
This repository contains two Jupyter notebooks (.ipynb files) demonstrating different data science techniques: Sentiment Analysis of Amazon reviews and Decision Tree/CHAID analysis for a funding prediction task.

1. Sentiment Analysis of Amazon Reviews (Sentiment_Analysis_.ipynb)
This notebook performs sentiment analysis on a dataset of Amazon reviews. It utilizes Natural Language Toolkit (NLTK) for text processing and sentiment scoring.

Key Features:
Data Loading and Exploration: Reads a CSV file (Amazon_reveiw.csv), displays its shape and head, and performs basic Exploratory Data Analysis (EDA) on the 'Score' column.

NLTK Basics: Demonstrates tokenization, part-of-speech tagging, and named entity recognition using NLTK.

VADER Sentiment Scoring: Applies NLTK's SentimentIntensityAnalyzer (VADER) to compute sentiment scores (negative, neutral, positive, and compound) for each review.

Visualization of Results: Presents the distribution of sentiment scores using bar plots, correlating them with the original Amazon star ratings.

Dependencies:
pandas

numpy

matplotlib

seaborn

nltk (with averaged_perceptron_tagger_eng, maxent_ne_chunker_tab, words, and vader_lexicon downloaded)

tqdm (for progress bars)

2. Decision Tree and CHAID Analysis (decision tree and chaid.ipynb)
This notebook focuses on building and evaluating a Decision Tree model, with an emphasis on data preparation and feature importance, for a funding prediction dataset.

Key Features:
Data Loading and Preprocessing: Loads a CSV file (Group1&2 - DecisionTree & CHAID.csv). It includes robust steps for:

Converting the 'Is Funded' column into a binary target variable.

Cleaning and converting financial columns (e.g., 'Total Funding', 'Annual Revenue') from string to float, handling currency symbols and commas.

Imputing missing values, specifically using the median for numerical features.

Label Encoding categorical variables such as 'SDG', 'Country', 'State', and 'City'.

Decision Tree Model:

Splits the data into training and testing sets.

Builds and trains a DecisionTreeClassifier.

Evaluates the model using accuracy score and a classification report.

Decision Tree Visualization: Generates a visual representation of the decision tree for better interpretability, with increased figure size, font size, and limited depth for readability.

Feature Importance: Calculates and displays the importance of each feature in the trained Decision Tree model.

Dependencies:
pandas

numpy

matplotlib

sklearn (specifically LabelEncoder, train_test_split, DecisionTreeClassifier, accuracy_score, classification_report, plot_tree, SimpleImputer)

Getting Started
Clone the repository:

git clone <repository_url>

Navigate to the project directory:

cd <project_directory>

Install the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn nltk tqdm

Download NLTK data (within the Sentiment_Analysis_.ipynb notebook):
The notebook includes commands to download necessary NLTK resources like averaged_perceptron_tagger_eng, maxent_ne_chunker_tab, words, and vader_lexicon. Run these cells when you first open the notebook.

Open the notebooks:

jupyter notebook

This will open Jupyter in your web browser, from where you can select and run either notebook.
