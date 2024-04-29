
| CS-688   | Web Minning and Graph Analysis       |
|----------|--------------------------------------|
| Name     | Paridhi Talwar                       |
| Date     | 04/19/2024                           |
| Course   | Spring'2024                          |
| Project  | Sentimental Analysis                 |

# Sentiment Analysis Project

This project aims to perform sentiment analysis on book reviews using natural language processing (NLP) techniques. The objective is to analyze the sentiment expressed in customer reviews for various books, visualize the sentiment distribution, explore sentiment trends over time, and identify the top books with positive and negative sentiment.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Dependencies](#dependencies)
6. [Data Preprocessing](#data-preprocessing)
7. [Sentiment Analysis](#sentiment-analysis)
8. [Visualization](#visualization)
9. [License](#license)

## Introduction
Sentiment analysis, also known as opinion mining, is the process of determining the sentiment expressed in text data. This project leverages Streamlit, a Python library for building interactive web applications, to create an intuitive interface for exploring sentiment analysis results. The application preprocesses the text data, performs sentiment analysis using NLTK and Scikit-learn libraries, and visualizes the results using various charts and graphs.

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/paridhitalwar/SentimentAnalysis_BookReviews.git
   ```

2. Navigate to the project directory:
   ```bash
   cd SentimentAnalysis_BookReviews
   ```

 3. Install the required Python packages:
    ```bash
    pip install streamlit scikit-learn numpy matplotlib nltk pandas seaborn
    ``` 

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage
Upon running the application, users can interact with the following features:
- **Upload CSV File**: Users can upload a CSV file containing book reviews for sentiment analysis.
- **Overall Sentiment Distribution**: Visualizes the distribution of sentiment categories (positive, negative, neutral) across all reviews.
- **Sentiment Score Distribution**: Displays a histogram of sentiment scores, showing the distribution of sentiment intensity.
- **Top Books with Positive/Negative Sentiment**: Lists the top 10 books with the highest positive and negative sentiment based on customer reviews.
- **Select Book for Detailed Analysis**: Allows users to select a specific book for detailed sentiment analysis, including sentiment distribution and sentiment scores over time (if date information is available).

## Project Structure
The project structure is organized as follows:
- `app.py`: Main Streamlit application file containing the code for preprocessing, sentiment analysis, and visualization.

## Dependencies
The project relies on the following Python libraries:
- Streamlit: For creating interactive web applications.
- Pandas: For data manipulation and analysis.
- NumPy: For numerical computing.
- NLTK (Natural Language Toolkit): For natural language processing tasks such as tokenization, lemmatization, and part-of-speech tagging.
- Scikit-learn: For machine learning algorithms and text vectorization.
- Matplotlib: For creating static, interactive, and animated visualizations in Python.
- Seaborn: For statistical data visualization.

## Data Preprocessing
- The text data is preprocessed to convert it into a suitable format for analysis.
- Preprocessing steps include lowercasing, removing special characters, stopwords, single characters, and numbers.
- Tokenization, lemmatization, and part-of-speech tagging are performed to extract meaningful features from the text.

## Sentiment Analysis
- Sentiment analysis is conducted using a combination of lexical and machine learning-based approaches.
- Lexical analysis involves assigning sentiment scores to words using sentiment lexicons such as SentiWordNet.
- Machine learning models, such as logistic regression, are trained on labeled data to predict sentiment based on textual features.

## Visualization
- Visualization plays a crucial role in understanding and interpreting sentiment analysis results.
- Charts and graphs, including bar charts, line charts, and histograms, are used to visualize sentiment distribution, sentiment scores over time, and other relevant insights.

## License
This project is licensed under the [MIT License](LICENSE).
