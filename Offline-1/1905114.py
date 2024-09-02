import pandas as pd

dataframe = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

dataframe.shape

dataframe.describe()

dataframe.isnull().sum()

dataframe[dataframe.duplicated(keep=False)]

# handle the null values with the mean of the column
dataframe.fillna(dataframe.select_dtypes(include=['number']).mean(), inplace=True)

dataframe.dropna(subset=['Attrition'], inplace=True)

dataframe.drop_duplicates(inplace=True)

features = dataframe.drop('Attrition', axis=1)
target = dataframe['Attrition']

# List of columns to convert to categorical
categorical_columns = [
    'BusinessTravel',
    'Department',
    'EducationField',
    'Gender',
    'JobRole',
    'MaritalStatus',
    'Over18',
    'OverTime'
]

# Convert each column to categorical type
for col in categorical_columns:
    features[col] = features[col].astype('category')

features.dtypes

features = pd.get_dummies(features)

# Convert only specific columns back to 0 and 1
boolean_columns = features.select_dtypes(include=['bool']).columns

# Convert the selected boolean columns to integers (0 and 1)
features[boolean_columns] = features[boolean_columns].astype(int)
features

features.dtypes

# Step 1: Take input from the user
#user_input = input("Press 0 for minmax, press 1 for standard input: ")

# Step 2: Convert the input to an integer
#number = int(user_input)

number = 0

# Step 3: Use if and else statements to check the input value
if number == 0:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features) #fit between 0 and 1
elif number == 1:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features) #It transforms the data so that it has a mean of 0 and a standard deviation of 1
features_normalized[0]     

features_df = pd.DataFrame(features_normalized, columns=features.columns) # scaled feature dataframe
target_df = pd.DataFrame(target, columns=['Attrition'])

# correlation analysis of features with target
# Convert 'Yes' to 1 and 'No' to 0 in the target column
target_df['Attrition'] = target_df['Attrition'].map({'Yes': 1, 'No': 0})
target_series = target_df['Attrition']
correlations = features_df.corrwith(target_series) #contribution of each column
correlations

top_20_correlations = correlations.abs().sort_values(ascending=False).head(20)
top_20_correlations

import matplotlib.pyplot as plt
import numpy as np


# Assuming 'correlations' is a Pandas Series with feature correlation values
# Sort correlations and get the top 20
top_20_features = correlations.abs().sort_values(ascending=False).head(20).index

# Loop through each of the top 20 features
for feature in top_20_features:
    # Separate the data based on the Attrition class
    class_0 = features_df[target_df["Attrition"] == 0]
    class_1 = features_df[target_df["Attrition"] == 1]

    # Plot 1D scatter plot
    plt.figure(figsize=(8, 4))
    plt.plot(class_0[feature], np.zeros_like(class_0[feature]), 'o', label='Attrition 0')
    plt.plot(class_1[feature], np.zeros_like(class_1[feature]), 'o', label='Attrition 1')

    plt.legend()
    plt.xlabel(feature)
    plt.title(f'1D Scatter Plot of {feature} by Attrition Classes')
    plt.show()