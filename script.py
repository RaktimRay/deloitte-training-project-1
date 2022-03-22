import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/rakray/Documents/Deloitte_Training/Day_2/AWS_architecture/deloitee-training-project-1/datasets/application_data.csv")
df_application_data = pd.DataFrame(data)
data = pd.read_csv("C:/Users/rakray/Documents/Deloitte_Training/Day_2/AWS_architecture/deloitee-training-project-1/datasets/previous_application.csv")
df_previous_application = pd.DataFrame(data)

# Main formula
acceptable_non_NAN_values_fraction = 0.5
df2 = df_application_data.dropna(axis='columns', how="any", thresh=(1-acceptable_non_NAN_values_fraction)*len(df_application_data.index))
df3 = df_previous_application.dropna(axis='columns', how="any", thresh=(1-acceptable_non_NAN_values_fraction)*len(df_previous_application.index))

print("acceptable NAN value percentage: "+str(acceptable_non_NAN_values_fraction*100)+"%")

print("columns needed to be kept from application_data for given acceptable NAN value fraction in a column:")
print(len(df2.columns))
print("columns needed to be dropped from application_data for given acceptable NAN value fraction in a column:")
print(len(df_application_data.columns) - len(df2.columns))
print("final dataframe for application_data:")
print(df2.head())

print("columns needed to be kept from previous_application for given acceptable NAN value fraction in a column:")
print(len(df3.columns))
print("columns needed to be dropped from previous_application for given acceptable NAN value fraction in a column:")
print(len(df_previous_application.columns) - len(df3.columns))
print("final dataframe for previous_application:")
print(df3.head())
