import numpy as np
import pandas as pd

calories_df = pd.read_csv('./calories.csv') # reading the csv files
exercise_df = pd.read_csv('./exercise.csv')

merged_df = pd.merge(calories_df,exercise_df, on='User_ID') # merging the files together based on user id
merged_df['Gender'] = merged_df['Gender'].map({'female': 0, 'male': 1}) # mapping male and female to 0 and 1

# Convert the features from a dataframe to a numpy array by extracting values
X = np.array(merged_df[['Gender','Age','Height','Weight','Heart_Rate','Body_Temp','Duration']])
y = np.array(merged_df['Calories'])

X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # Adding a column of ones as the intercept
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) # Linear regression using the normal equation

print(f"Intercept: {theta[0]:.2f}\n") # Printing the intercept and all the slopes of best-fitting lines
print("Coefficients for features:")
features_names = ['Gender','Age','Height','Weight','Heart_Rate','Body_Temp','Duration']
for i in range(1, len(theta)):
    print(f"{features_names[i-1]}: {theta[i]:.2f}")

def predict_calories(features, theta):
    features_with_intercept = np.concatenate(([1], features)) # Again adding a column of ones
    return np.dot(features_with_intercept, theta) # Predicting using the theta matrix

# Taking user inputs
gender = {"female": 0, "male": 1}[input("\nEnter gender (male/female): ").lower()]
age = float(input("Enter age (in years) : "))
height = float(input("Enter height (in cm) : "))
weight = float(input("Enter weight (in kg) : "))
heart_rate = float(input("Enter heart rate (in beats per minute) : "))
body_temp = float(input("Enter body temperature (in Celsius) : "))
duration = float(input("Enter duration of exercise (in minutes) : "))
features = [gender, age, height, weight, heart_rate, body_temp, duration]

print(f"\nPredicted calories burnt: {predict_calories(features, theta):.2f}") # Calculating the calories

predicted_values = np.dot(X, theta)
total_sum_squares = np.sum((y - np.mean(y))**2)
residual_sum_squares = np.sum((y - predicted_values)**2)
r_squared = 1 - (residual_sum_squares / total_sum_squares)

print(f"R^2 score: {r_squared:.2f}")