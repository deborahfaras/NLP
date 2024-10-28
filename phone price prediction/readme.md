# Price Prediction using Random Forest Classifier

## Project Overview

This project aims to predict the prices of smartphones based on their titles using a Random Forest Regressor. The dataset contains information such as the title, rating, reviews, price, image URL, and product URL of various smartphones. The project involves data preprocessing, feature extraction using TF-IDF vectorization, model training, hyperparameter tuning, and evaluation.

## Dataset

The dataset used in this project is `dataset.csv`, which contains the following columns:

- `Title`: The name of the smartphone.
- `Rating`: The rating of the smartphone.
- `Reviews`: The number of reviews the smartphone has received.
- `Price`: The price of the smartphone.
- `Image URL`: The URL of the smartphone's image.
- `Product URL`: The URL of the smartphone's product page.

## Project Structure

- `preprocess_text(text)`: Function to preprocess the text by converting it to lowercase and removing special characters.
- Data preprocessing applied to the `Title` column to create a `Processed_Title` column.
- TF-IDF Vectorization applied to `Processed_Title` to create features for the model.
- `convert_price(price)`: Function to clean and convert the price to a numeric format.
- Random Forest Regressor model training and hyperparameter tuning using GridSearchCV.
- Evaluation of the model using Mean Squared Error (MSE).
- Visualization of actual vs predicted prices using a scatter plot.

### Prerequisites

- Python 3.6+
- pandas
- scikit-learn
- matplotlib
- wordcloud
- nltk

