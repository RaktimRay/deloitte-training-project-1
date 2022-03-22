import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

def outlier_treatment(datacolumn):
 sorted(datacolumn)
 Q1,Q3 = np.percentile(datacolumn , [25,75])
 IQR = Q3 - Q1
 lower_range = Q1 - (1.5 * IQR)
 upper_range = Q3 + (1.5 * IQR)
 return lower_range,upper_range

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
for col in df_application_data_2:
    if (df_application_data_2[col].dtype == "int64" or df_application_data_2[col].dtype == "float64"):
        df_application_data_2[col] = df_application_data_2[col].fillna(df_application_data_2[col].median())
    elif (df_application_data_2[col].dtype == "object"):
        df_application_data_2[col] = df_application_data_2[col].fillna(df_application_data_2[col].mode().iloc[0])

for col in df_previous_application_2:
    if (df_previous_application_2[col].dtype == "int64" or df_previous_application_2[col].dtype == "float64"):
        df_previous_application_2[col] = df_previous_application_2[col].fillna(df_previous_application_2[col].median())
    elif (df_previous_application_2[col].dtype == "object"):
        df_previous_application_2[col] = df_previous_application_2[col].fillna(df_previous_application_2[col].mode().iloc[0])
    
# Boxplot
# df_application_data_2.boxplot(column="ad_amt_income_total_ct")
fig= plt.figure(figsize=(10,7)) 
# Creating plot
plt.boxplot(df_previous_application_2["ad_amt_credit_ct"])
# show plot
plt.show()


# Outlier
lowerbound,upperbound = outlier_treatment(df_previous_application_2.ad_amt_credit_ct)
# check outlier for selected column
df_previous_application_2[(df_previous_application_2.ad_amt_credit_ct < lowerbound) | (df_previous_application_2.ad_amt_credit_ct > upperbound)]
# Remove outlier
df_previous_application_2.drop(df_previous_application_2[ (df_previous_application_2.ad_amt_credit_ct > upperbound) | (df_previous_application_2.ad_amt_credit_ct < lowerbound) ].index , inplace=True)

print(df_previous_application_2.describe())

# Boxplot outlier removed
# df_application_data_2.boxplot(column="ad_amt_income_total_ct")
fig= plt.figure(figsize=(10,7)) 
# Creating plot
plt.boxplot(df_previous_application_2["ad_amt_credit_ct"])
# show plot
plt.show()


#shown for previous application_data.csv file, same can be done for the other csv file following the same steps