import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("C:/Users/rakray/Documents/Deloitte_Training/Day_2/AWS_architecture/deloitee-training-project-1/datasets/application_data.csv")
df_application_data = pd.DataFrame(data)
data = pd.read_csv("C:/Users/rakray/Documents/Deloitte_Training/Day_2/AWS_architecture/deloitee-training-project-1/datasets/previous_application.csv")
df_previous_application = pd.DataFrame(data)

# Missing value removal
acceptable_non_NAN_values_fraction = 0.5
df_application_data_2 = df_application_data.dropna(axis='columns', how="any", thresh=(1-acceptable_non_NAN_values_fraction)*len(df_application_data.index))
df_previous_application_2 = df_previous_application.dropna(axis='columns', how="any", thresh=(1-acceptable_non_NAN_values_fraction)*len(df_previous_application.index))

# NAN value replacement
def NAN_value_replacement(dataframe):
    for col in dataframe:
        if (dataframe[col].dtype == "int64" or dataframe[col].dtype == "float64"):
            dataframe[col] = dataframe[col].fillna(dataframe[col].median())
        elif (dataframe[col].dtype == "object"):
            dataframe[col] = dataframe[col].fillna(dataframe[col].mode().iloc[0])
    return dataframe

df_application_data_2 = NAN_value_replacement(df_application_data_2)
df_previous_application_2 = NAN_value_replacement(df_previous_application_2)
    
# Boxplot
def Boxplot(dataframe, column):
    fig= plt.figure(figsize=(10,7))
    plt.boxplot(dataframe[column])
    plt.show()

# Initial Boxplot
Boxplot(df_application_data_2, "ad_amt_credit_ct") #current data
Boxplot(df_previous_application_2, "ad_amt_credit_ct") #previous data

# Getting numerical only Dataframe from full Dataframe
def numerical_df(df):
    numerical = df.select_dtypes(exclude='object')
    return numerical

# gettting numerical dfs
numerical_df_previous_application = numerical_df(df_previous_application_2)
numerical_df_application_data = numerical_df(df_application_data_2)


# Outlier removing function
def outlier_removal(df, numerical_only_df):
    Q1 = numerical_only_df.quantile(0.25)
    Q3 = numerical_only_df.quantile(0.75)
    IQR = Q3 - Q1

    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df

# Removing outliers
df_previous_application_2 = outlier_removal(df_previous_application_2, numerical_df_previous_application)
df_application_data_2 = outlier_removal(df_application_data_2, numerical_df_application_data)

# show modified dataframes with outliers removed
print(df_previous_application_2.describe())
print(df_previous_application_2.shape)
print(df_application_data_2.describe())
print(df_application_data_2.shape)

# Boxplot outlier removed
Boxplot(df_application_data_2, "ad_amt_credit_ct") #current data
Boxplot(df_previous_application_2, "ad_amt_credit_ct") #previous data