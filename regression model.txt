import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from google.colab import files
uploaded = files.upload()

# Load the data from all three JSON files
data_2012 = pd.read_json("2012 Olympic Medals.json")
data_2016 = pd.read_json("2016 Olympic Medals.json")
data_2021 = pd.read_json("2021 Olympic Medals.json")

# Standardize column names before concatenation
data_2021.rename(columns={'Country': 'Country Name'}, inplace=True) 

# Concatenate the data from all three years
data = pd.concat([data_2012, data_2016, data_2021])

# Calculate 'Total Medals' for the combined dataframe
data['Total Medals'] = data['Gold Medals'] + data['Silver Medals'] + data['Bronze Medals']

# Sort the data by 'Total Medals' and select the top 10
top_10 = data.sort_values('Total Medals', ascending=False).head(10)

# Define a function to train a regression model and make predictions
def predict_medals(country):
    # Check if the country is in the top 10
    if country not in top_10['Country Name'].values:
        return f"{country} is not in the top 10 countries."

    # For each medal type
    for medal_type in ['Gold Medals', 'Silver Medals', 'Bronze Medals']:
        # Prepare the data
        X = top_10.drop([medal_type, 'Country Name'], axis=1)
        y = top_10[medal_type]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the model
        model = LinearRegression()

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions for the given country
        prediction = model.predict([top_10[top_10['Country Name'] == country].drop([medal_type, 'Country Name'], axis=1).iloc[0]])

        # Print the prediction
        print(f"Predicted number of {medal_type} for {country} in 2024: {prediction[0]}")

# Use the function to predict the number of medals for a given country
predict_medals('United States') # Use 'United States' as it appears in the 'Country Name' column
