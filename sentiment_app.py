import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')

st.write("""
# Sentiment Analysis
""")


# Load data
books_df = pd.read_csv('C:/Users/parid/OneDrive/Desktop/STUDY  BU/CS688 Web Mining and Analytics/sentiment/Top-100 Trending Books.csv')
reviews_df = pd.read_csv('C:/Users/parid/OneDrive/Desktop/STUDY  BU/CS688 Web Mining and Analytics/sentiment/customer reviews.csv')

# Rename the column in the second dataset
reviews_df.rename(columns={'book name': 'book title'}, inplace=True)
# Remove the 'Sno' column from reviews_df
reviews_df.drop(columns=['Sno'], inplace=True)

# Merge datasets on 'book title'
merged_df = pd.merge(books_df, reviews_df, on='book title')

# Function to preprocess Reviews data
def preprocess_Reviews_data(merged_df,name):
    # Proprocessing the data
    merged_df[name]=merged_df[name].str.lower()
    # Code to remove the Special characters from the text 
    merged_df[name]=merged_df[name].apply(lambda x:' '.join(re.findall(r'\w+', str(x))))
    # Code to substitute the multiple spaces with single spaces
    merged_df[name]=merged_df[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    # Code to remove all the single characters in the text
    merged_df[name]=merged_df[name].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))

# Function to tokenize and remove the stopwords    
def rem_stopwords_tokenize(merged_df,name):
      
    def getting(sen):
        example_sent = sen
        
        filtered_sentence = [] 

        stop_words = set(stopwords.words('english')) 

        word_tokens = word_tokenize(example_sent) 
        
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        
        return filtered_sentence
    # Using "getting(sen)" function to append edited sentence to data
    x=[]
    for i in merged_df[name].values:
        x.append(getting(i))
    merged_df[name]=x
    
lemmatizer = WordNetLemmatizer()
def Lemmatization(merged_df,name):
    def getting2(sen):
        example = sen
        output_sentence =[]
        word_tokens2 = word_tokenize(example)
        lemmatized_output = [lemmatizer.lemmatize(w) for w in word_tokens2]
        
        # Remove characters which have length less than 2  
        without_single_chr = [word for word in lemmatized_output if len(word) > 2]
        # Remove numbers
        cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]
        
        return cleaned_data_title
    # Using "getting2(sen)" function to append edited sentence to data
    x=[]
    for i in merged_df[name].values:
        x.append(getting2(i))
    merged_df[name]=x

def make_sentences(merged_df,name):
    merged_df[name]=merged_df[name].apply(lambda x:' '.join([i+' ' for i in x]))
    # Removing double spaces if created
    merged_df[name]=merged_df[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    
# Using the preprocessing function to preprocess the data
preprocess_Reviews_data(merged_df,'review description')
# Using tokenizer and removing the stopwords
rem_stopwords_tokenize(merged_df,'review description')
# Converting all the texts back to sentences
make_sentences(merged_df,'review description')

#Edits After Lemmatization
final_Edit = merged_df['review description'].copy()
merged_df["After_lemmatization"] = final_Edit

# Using the Lemmatization function to lemmatize thedata
Lemmatization(merged_df,'After_lemmatization')
# Converting all the texts back to sentences
make_sentences(merged_df,'After_lemmatization')

pos=neg=obj=count=0

postagging = []

for review in merged_df['After_lemmatization']:
    list = word_tokenize(review)
    postagging.append(nltk.pos_tag(list))

merged_df['pos_tags'] = postagging

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


# Returns list of pos-neg and objective score. But returns empty list if not present in senti wordnet.
def get_sentiment(word,tag):
    wn_tag = penn_to_wn(tag)
    
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    #Lemmatization
    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    #Synset is a special kind of a simple interface that is present in NLTK to look up words in WordNet. 
    #Synset instances are the groupings of synonymous words that express the same concept. 
    #Some of the words have only one Synset and some have several.
    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())
#  self.obj_score = 1.0 - (self.pos_score + self.neg_score)

    return [synset.name(), swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]

    pos=neg=obj=count=0
    
    ###################################################################################
senti_score = []

for pos_val in merged_df['pos_tags']:
    senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
    for score in senti_val:
        try:
            pos = pos + score[1]  #positive score is stored at 2nd position
            neg = neg + score[2]  #negative score is stored at 3rd position
        except:
            continue
    senti_score.append(pos - neg)
    pos=neg=0    
    
merged_df['senti_score'] = senti_score
print(merged_df['senti_score'])

print(merged_df.head(5))

overall=[]
for i in range(len(merged_df)):
    if merged_df['senti_score'][i]> 0.5:
        overall.append('Positive')
    elif merged_df['senti_score'][i]< 0.5:
        overall.append('Negative')
    else:
        overall.append('Neutral')
merged_df['Overall Sentiment']=overall
merged_df.to_csv('sentiment_analysis_output.csv', index=False)

# Feature Engineering
X = merged_df['After_lemmatization']
y = merged_df['Overall Sentiment']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing text data
vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the number of features
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Sidebar
st.sidebar.title('Upload CSV File')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])

# Main content
st.title('Sentiment Analysis Results')

if uploaded_file is not None:
    new_reviews_df = pd.read_csv(uploaded_file)

    # Preprocess the new data similarly to the training data
    preprocess_Reviews_data(new_reviews_df, 'review description')
    rem_stopwords_tokenize(new_reviews_df, 'review description')
    make_sentences(new_reviews_df, 'review description')
    new_reviews_df['After_lemmatization'] = new_reviews_df['review description'].copy()
    Lemmatization(new_reviews_df, 'After_lemmatization')
    make_sentences(new_reviews_df, 'After_lemmatization')

    # POS tagging for the new reviews
    postagging_new = []

    for review in new_reviews_df['After_lemmatization']:
        list = word_tokenize(review)
        postagging_new.append(nltk.pos_tag(list))

    new_reviews_df['pos_tags'] = postagging_new

    # Calculate sentiment scores for the new reviews
    senti_score = []

    for pos_val in new_reviews_df['pos_tags']:
        senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
        for score in senti_val:
            try:
                pos = pos + score[1]  #positive score is stored at 2nd position
                neg = neg + score[2]  #negative score is stored at 3rd position
            except:
                continue
        senti_score.append(pos - neg)
        pos=neg=0    
        
    new_reviews_df['senti_score'] = senti_score
    # Determine the overall sentiment for each review
    overall_new = []

    for score in new_reviews_df['senti_score']:
        if score > 0.5:
            overall_new.append('Positive')
        elif score < 0.5:
            overall_new.append('Negative')
        else:
            overall_new.append('Neutral')

    new_reviews_df['Overall Sentiment'] = overall_new

    
    # Save the updated DataFrame to a new CSV file
    new_reviews_df.to_csv('sentiment_analysis.csv', index=False)

    # Vectorize the text using the same vectorizer
    new_reviews_vec = vectorizer.transform(new_reviews_df['After_lemmatization'])

    # Predict sentiment
    new_reviews_pred = model.predict(new_reviews_vec)
    print("Predicted Sentiment for New Reviews:")
    print(new_reviews_pred)
    # Display overall sentiment count
    st.subheader("Overall Sentiment Distribution")
    overall_sentiment_counts = new_reviews_df['Overall Sentiment'].value_counts()
    st.bar_chart(overall_sentiment_counts)

    # Display sentiment score distribution
    st.subheader("Sentiment Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(new_reviews_df['senti_score'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # Display top 10 books with positive sentiment
    st.subheader("Top 10 Books with Positive Sentiment")
    top_positive_books = new_reviews_df[new_reviews_df['Overall Sentiment'] == 'Positive']['book title'].value_counts().head(10)
    st.write(top_positive_books)

    # Display top 10 books with negative sentiment
    st.subheader("Top 10 Books with Negative Sentiment")
    top_negative_books = new_reviews_df[new_reviews_df['Overall Sentiment'] == 'Negative']['book title'].value_counts().head(10)
    st.write(top_negative_books)

    # Display dropdown for selecting book
    selected_book = st.sidebar.selectbox('Select a Book:', new_reviews_df['book title'].unique())
    book_info = new_reviews_df[new_reviews_df['book title'] == selected_book].iloc[0]

    # Display book information
    st.sidebar.subheader("Book Information")
    st.sidebar.write("**Title:**", book_info['book title'])
    st.sidebar.write("**Author:**", book_info['author'])
    st.sidebar.write("**Rating:**", book_info['rating'])

    # Display sentiment graph for selected book
    st.subheader("Sentiment Analysis for Selected Book")
    selected_book_df = new_reviews_df[new_reviews_df['book title'] == selected_book]
    sentiment_counts = selected_book_df['Overall Sentiment'].value_counts()
    st.bar_chart(sentiment_counts)
    
    # Display sentiment scores over time for the selected book if 'date' column is present
    if 'date' in selected_book_df.columns:
        st.subheader("Sentiment Scores Over Time for Selected Book")
        selected_book_df['date'] = pd.to_datetime(selected_book_df['date'], format='%d-%m-%Y')
        sentiment_over_time = selected_book_df.groupby('date')['senti_score'].mean().reset_index()
        st.line_chart(sentiment_over_time.set_index('date'))

