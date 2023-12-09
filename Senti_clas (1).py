import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import re
import streamlit as st
import pickle

# Load the dataset
file_path = 'Tweets.csv'
dataset = pd.read_csv(file_path)

# Function to clean and preprocess text data
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-letters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lower case
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply the cleaning function to the text column
dataset['clean_text'] = dataset['text'].astype(str).apply(clean_text)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['clean_text'], dataset['sentiment'], test_size=0.2, random_state=42)

# Creating a pipeline with TfidfVectorizer and Multinomial Naive Bayes classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB()),
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
predictions = pipeline.predict(X_test)
report = classification_report(y_test, predictions)

# Save the trained model to a pickle file
pickle_filename = 'sentiment_analysis_model.pkl'
with open(pickle_filename, 'wb') as file:
    pickle.dump(pipeline, file)

# Streamlit application layout (this part should be in a separate file for the Streamlit app)
st.title("Tweet Sentiment Analysis")
st.write("Enter a tweet to predict its sentiment.")

# User input
user_input = st.text_area("Tweet Text", "")

# Load the model (this should be outside the 'if' condition to load the model when the app starts)
with open(pickle_filename, 'rb') as file:
    model = pickle.load(file)

# Predict and display the result
if st.button("Predict Sentiment"):
    prediction = model.predict([user_input])[0]
    st.write(f"Predicted Sentiment: {prediction}")