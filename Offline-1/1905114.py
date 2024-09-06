# importing pandas csv library:
import pandas as pd

# stroing csv in variable:
dataframe = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Showing number of attributes:
dataframe.shape

# showing the statistics of the dataset: 
dataframe.describe()

# number of missing values in the dataset:
dataframe.isnull().sum()

# number of duplicate values in the dataset
dataframe[dataframe.duplicated(keep=False)]

# dropping row if target is null
dataframe.dropna(subset=['Attrition'], inplace=True)

# replacing any null number values with columnwise mean:
dataframe.fillna(dataframe.select_dtypes(include=['number']).mean(), inplace=True)

# replacing any null object values with columnwise most frequent:
for column in dataframe.select_dtypes(include=['object']).columns:
    most_frequent = dataframe[column].mode()[0]
    dataframe[column].fillna(most_frequent, inplace=True)

# keeping one copy of row if duplicate row found
dataframe.drop_duplicates(inplace=True)

# splitting data between Features and Labels
Features = dataframe.drop('Attrition', axis=1)
Labels = dataframe['Attrition']

#using label encoding for some binary value columns
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Iterate over the object columns and apply label encoding to columns with two unique values
for col in Features.select_dtypes(include=['object']).columns:
    if Features[col].nunique() == 2:  # Check if the column has only two unique values
        Features[col] = label_encoder.fit_transform(Features[col])

# showing data types of Features
Features.dtypes

# List of columns to convert to categorical
categorical_columns = Features.select_dtypes(include=['object']).columns.tolist()

# Convert non numeric columns to categorical type
for col in categorical_columns:
    Features[col] = Features[col].astype('category')

# showing data types of Features
Features.dtypes

# One hot encoding
Features = pd.get_dummies(Features)

# Convert only boolean columns back to 0 and 1
boolean_columns = Features.select_dtypes(include=['bool']).columns

# Convert the selected boolean columns to integers (0 and 1)
Features[boolean_columns] = Features[boolean_columns].astype(int)
Features

# showing features datatypes
Features.dtypes

# scaling type
scaling_type = "minmax"
#scaling_type = "standard"

def scaling(Features, scaling_type):
    # Identify binary columns (columns with only two unique values)
    binary_columns = [col for col in Features.columns if len(Features[col].unique()) == 2]
    
    # Separate binary and non-binary columns
    non_binary_columns = Features.columns.difference(binary_columns)
    
    if scaling_type == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif scaling_type == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

    # Apply scaling only to non-binary columns
    Features_scaled = Features.copy()
    Features_scaled[non_binary_columns] = scaler.fit_transform(Features[non_binary_columns])
    
    return Features_scaled

features_normalized = scaling(Features,scaling_type)

# features and labels are put into dataframe
features_df = pd.DataFrame(features_normalized, columns=Features.columns) # scaled feature dataframe
labels_df = pd.DataFrame(Labels, columns=['Attrition'])

# correlation analysis of features with target
# Convert 'Yes' to 1 and 'No' to 0 in the target column
labels_df['Attrition'] = labels_df['Attrition'].map({'Yes': 1, 'No': 0})
target_series = labels_df['Attrition']
correlations = features_df.corrwith(target_series) #contribution of each column
correlations

top_20_correlations = correlations.abs().sort_values(ascending=False).head(20)
top_20_correlations

import matplotlib.pyplot as plt
import numpy as np

# Assuming 'correlations' is a Pandas Series with feature correlation values
# Sort correlations and get the top 20
top_20_features = correlations.abs().sort_values(ascending=False).head(20).index

#Importnecessarylibraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


selected_features = Features[top_20_features]
features_df = pd.DataFrame(features_normalized[selected_features.columns], columns=selected_features.columns)
#features_df = pd.DataFrame(features_normalized, columns=Features.columns) # scaled feature dataframe

X_train,X_test,y_train,y_test=train_test_split(features_df,labels_df, test_size=0.2,random_state=42)
#Step2: InitializetheLogisticRegressionclassifier
clf=LogisticRegression()
#Step3:Traintheclassifieronthetrainingdata
clf.fit(X_train,y_train)
#Step4:Makepredictionsonthetestset
y_pred=clf.predict(X_test)
#Step5:Evaluatetheclassifier'sperformance
accuracy=accuracy_score(y_test,y_pred)
print(f"AccuracyofLogisticRegressionclassifier:{accuracy:.2f}")