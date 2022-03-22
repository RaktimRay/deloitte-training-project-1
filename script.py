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

Boxplot(df_application_data_2, "ad_amt_credit_ct") #current data
Boxplot(df_previous_application_2, "ad_amt_credit_ct") #previous data

# Outlier
def outlier_treatment(datacolumn):
 sorted(datacolumn)
 Q1,Q3 = np.percentile(datacolumn , [25,75])
 IQR = Q3 - Q1
 lower_range = Q1 - (1.5 * IQR)
 upper_range = Q3 + (1.5 * IQR)
 return lower_range,upper_range

# previous application
lowerbound,upperbound = outlier_treatment(df_previous_application_2.ad_amt_credit_ct)
# check outlier for selected column
df_previous_application_2[(df_previous_application_2.ad_amt_credit_ct < lowerbound) | (df_previous_application_2.ad_amt_credit_ct > upperbound)]
# Remove outlier
df_previous_application_2.drop(df_previous_application_2[ (df_previous_application_2.ad_amt_credit_ct > upperbound) | (df_previous_application_2.ad_amt_credit_ct < lowerbound) ].index , inplace=True)
print(df_previous_application_2.describe())

# current application
lowerbound,upperbound = outlier_treatment(df_application_data_2.ad_amt_credit_ct)
# check outlier for selected column
df_application_data_2[(df_application_data_2.ad_amt_credit_ct < lowerbound) | (df_application_data_2.ad_amt_credit_ct > upperbound)]
# Remove outlier
df_application_data_2.drop(df_application_data_2[ (df_application_data_2.ad_amt_credit_ct > upperbound) | (df_application_data_2.ad_amt_credit_ct < lowerbound) ].index , inplace=True)
print(df_application_data_2.describe())

# Boxplot outlier removed
Boxplot(df_application_data_2, "ad_amt_credit_ct") #current data
Boxplot(df_previous_application_2, "ad_amt_credit_ct") #previous data