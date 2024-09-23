import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model (LinearRegression is used as an example)
model = LinearRegression()
model.fit(X_train, y_train)

# Create the Streamlit app
def main():
    st.title("California Housing Price Prediction")

    # Input fields for user features (replace with relevant features from X)
    longitude = st.number_input("Longitude", min_value=float(X[:, 0].min()), max_value=float(X[:, 0].max()))
    latitude = st.number_input("Latitude", min_value=float(X[:, 1].min()), max_value=float(X[:, 1].max()))
    # Add other relevant features as needed

    # Create a button for prediction
    if st.button("Predict Price"):
        # Create a DataFrame with user input (using only selected features)
        user_input = pd.DataFrame([[longitude, latitude]], columns=["longitude", "latitude"])

        # Make prediction
        prediction = model.predict(user_input)[0]

        # Display prediction
        st.write("Predicted Price:", round(prediction, 2))

if __name__ == '__main__':
    main()