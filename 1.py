import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Assume you've collected and preprocessed data
X = pd.Series(['positive review', 'negative review', 'neutral review'])
y = pd.Series([1, 0, 2])

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_vectorized, y)

# Streamlit app
st.title('Sentiment Analysis')

text_input = st.text_input('Enter a text:')

if st.button('Analyze'):
    vectorized_text = vectorizer.transform([text_input])
    prediction = model.predict(vectorized_text)[0]

    if prediction == 1:
        st.success('Positive sentiment')
    elif prediction == 0:
        st.error('Negative sentiment')
    else:
        st.info('Neutral sentiment')