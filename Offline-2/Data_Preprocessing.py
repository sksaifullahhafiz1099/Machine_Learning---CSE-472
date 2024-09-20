# importing pandas csv library:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Data_Prep:
    def __init__(self,Target_column):
        self.Target_column = Target_column

    def data_import(self,file_name):
        # stroing csv in variable:
        dataframe = pd.read_csv(file_name)

        # number of duplicate values in the dataset
        dataframe[dataframe.duplicated(keep=False)]

        #return dataframe
        return dataframe

    def data_cleaning(self,dataframe):
        # dropping row if target is null
        dataframe.dropna(subset=[self.Target_column], inplace=True)

        # replacing any null number values with columnwise mean:
        dataframe.fillna(dataframe.select_dtypes(include=['number']).mean(), inplace=True)

        # replacing any null object values with columnwise most frequent:
        for column in dataframe.select_dtypes(include=['object']).columns:
            most_frequent = dataframe[column].mode()[0]
            dataframe[column].fillna(most_frequent, inplace=True)

        # keeping one copy of row if duplicate row found
        dataframe.drop_duplicates(inplace=True)
        
        #return dataframe
        return dataframe

    def input_output_feature_creation(self,dataframe):
        # splitting data between Features and Labels
        Features = dataframe.drop(self.Target_column, axis=1)
        Labels = dataframe[self.Target_column]
    
        #return
        return Features,Labels

    def numeric_conversion(self,Features):
        #using label encoding for some binary value columns
        from sklearn.preprocessing import LabelEncoder

        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        # Iterate over the object columns and apply label encoding to columns with two unique values
        for col in Features.select_dtypes(include=['object']).columns:
            if Features[col].nunique() == 2:  # Check if the column has only two unique values
                Features[col] = label_encoder.fit_transform(Features[col])

        # List of columns to convert to categorical
        categorical_columns = Features.select_dtypes(include=['object']).columns.tolist()

        # Convert non numeric columns to categorical type
        for col in categorical_columns:
            Features[col] = Features[col].astype('category')

        # One hot encoding
        Features = pd.get_dummies(Features)

        # Convert only boolean columns back to 0 and 1
        boolean_columns = Features.select_dtypes(include=['bool']).columns

        # Convert the selected boolean columns to integers (0 and 1)
        Features[boolean_columns] = Features[boolean_columns].astype(int)

        #return
        return Features

    def scaling(self,Features, scaling_type):
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

    def correlation_analysis(self,features_normalized,Features,Labels):
        # features and labels are put into dataframe
        features_df = pd.DataFrame(features_normalized, columns=Features.columns) # scaled feature dataframe
        labels_df = pd.DataFrame(Labels, columns=[self.Target_column])

        # correlation analysis of features with target
        # Convert 'Yes' to 1 and 'No' to 0 in the target column
        labels_df[self.Target_column] = labels_df[self.Target_column].map({'Yes': 1, 'No': 0})
        target_series = labels_df[self.Target_column]
        correlations = features_df.corrwith(target_series) #contribution of each column

        #selecting top 20 correlations
        top_20_correlations = correlations.abs().sort_values(ascending=False).head(20)

        return top_20_correlations,features_df,labels_df

    def show_plot(self,top_20_correlations,features_df,labels_df):
        # Assuming 'correlations' is a Pandas Series with feature correlation values
        # Sort correlations and get the top 20
        top_20_features = top_20_correlations.index

        # Loop through each of the top 20 features
        for feature in top_20_features:
            # Separate the data based on the Attrition class
            class_0 = features_df[labels_df[self.Target_column] == 0]
            class_1 = features_df[labels_df[self.Target_column] == 1]

            # Plot 1D scatter plot
            plt.figure(figsize=(8, 4))
            plt.plot(class_0[feature], np.zeros_like(class_0[feature]), 'o', label=self.Target_column+' 0')
            plt.plot(class_1[feature], np.zeros_like(class_1[feature]), 'o', label=self.Target_column+' 1')

            plt.legend()
            plt.xlabel(feature)
            plt.title(f'1D Scatter Plot of {feature} by {self.Target_column} Classes')
            plt.show()