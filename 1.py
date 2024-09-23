import streamlit as st
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd

# Load the iris dataset
iris = load_iris()
X = iris.data
df = pd.DataFrame(X, columns=iris.feature_names)

# Sidebar for user input
st.sidebar.header('User Input Parameters')
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
    sepal_width = st.sidebar.slider('Sepal width', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
    petal_length = st.sidebar.slider('Petal length', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
    petal_width = st.sidebar.slider('Petal width', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Combine user input with the entire dataset
df = pd.concat([input_df, df], axis=0)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)
df['cluster'] = kmeans.predict(df)

# Display the cluster of the user input
st.subheader('Cluster Prediction')
st.write('The input data belongs to cluster:', kmeans.predict(input_df)[0])

# Display the entire dataframe with cluster labels
st.subheader('Clustered Data')
st.write(df)