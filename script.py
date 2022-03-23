import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from skfeature.function.similarity_based import fisher_score
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("C:/Users/rakray/Documents/Deloitte_Training/Code/deloitte-training-project-1/datasets/application_data.csv")
df_application_data = pd.DataFrame(data)
data = pd.read_csv("C:/Users/rakray/Documents/Deloitte_Training/Code/deloitte-training-project-1/datasets/previous_application.csv")
df_previous_application = pd.DataFrame(data)

# print("Shape Original - previous_application.csv")
# print(df_previous_application.shape)
# print("Shape Original - application_data.csv")
# print(df_application_data.shape)

# Missing value removal
acceptable_non_NAN_values_fraction = 0.5
df_application_data_2 = df_application_data.dropna(axis='columns', how="any", thresh=(1-acceptable_non_NAN_values_fraction)*len(df_application_data.index))
df_previous_application_2 = df_previous_application.dropna(axis='columns', how="any", thresh=(1-acceptable_non_NAN_values_fraction)*len(df_previous_application.index))

# print("Shape after dropping - previous_application.csv")
# print(df_previous_application_2.shape)
# print("Shape after dropping - application_data.csv")
# print(df_application_data_2.shape)

# Shapiro test

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

# print("Shape after NAN value replacement - previous_application.csv")
# print(df_previous_application_2.shape)
# print("Shape after NAN value replacement - application_data.csv")
# print(df_application_data_2.shape)
    
# Boxplot
def Boxplot(dataframe, column):
    plt.figure(figsize=(10,7))
    plt.title(column)
    plt.boxplot(dataframe[column])
    plt.show()

# Initial Boxplot
# Boxplot(df_application_data_2, "ad_amt_credit_ct") #current data
# Boxplot(df_previous_application_2, "ad_amt_credit_ct") #previous data

# EDIT: might not need numerical dataframe
# Getting numerical only Dataframe from full Dataframe
def numerical_df(df):
    numerical = df.select_dtypes(exclude='object')
    return numerical
# gettting numerical dfs
numerical_df_previous_application = numerical_df(df_previous_application_2)
numerical_df_application_data = numerical_df(df_application_data_2)


# Outlier removing function
def outlier_removal(df, numerical_only_df): # Passing extra argument just for testing purpose so less code changes required later
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    df_final = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_final

# Removing outliers
df_previous_application_2 = outlier_removal(df_previous_application_2, numerical_df_previous_application)
df_application_data_2 = outlier_removal(df_application_data_2, numerical_df_application_data)

# Final Shape
# print("Shape without outlier - previous_application.csv")
# print(df_previous_application_2.shape)
# print("Shape without outlier - application_data.csv")
# print(df_application_data_2.shape)

# Boxplot outlier removed
# Boxplot(df_application_data_2, "ad_amt_credit_ct") #current data
# Boxplot(df_previous_application_2, "ad_amt_credit_ct") #previous data

# printing column
# print(abs(df_previous_application_2["ad_days_decision_ct"].head(10)))
# print(abs(df_previous_application_2["ad_days_decision_ct"].max()))
# print(abs(df_previous_application_2["ad_days_decision_ct"].min()))

# printing numerical columns
# print("previous_application numerical columns are:")
# for col in numerical_df_previous_application.columns:
#     print(col)

# numerical to categorical
df_previous_application_2["ad_MONTHS_decision_ct"] = abs(df_previous_application_2["ad_days_decision_ct"])/30
bins = [0,1,2,3,4,5,6,7,8,9,np.inf]
slots = ["0-1","1-2","2-3","3-4","4-5","5-6","6-7","7-8","8-9","Above 9"]
df_previous_application_2["ad_MONTHS_decision_ct"] = pd.cut(df_previous_application_2["ad_MONTHS_decision_ct"], bins=bins, labels=slots)


# print(df_previous_application_2["ad_name_contract_status_ct"].value_counts())

# Encoding
def encoder(dataframe, column):
    label_encoder = preprocessing.LabelEncoder()
    dataframe[column]= label_encoder.fit_transform(dataframe[column])
    # print(dataframe[column].value_counts())

encoder(df_previous_application_2, "ad_name_contract_status_ct")

# Feature Selection - Correleation Coefficient
def corr_co(name, dataframe, thresh, plot_visibility):
    print("\nThis is correlation coeeficient function for "+name)
    corr_matrix = dataframe.corr()
    if (plot_visibility):
        sns.heatmap(corr_matrix,annot=True,cmap=plt.cm.CMRmap_r)
        plt.show()

    coll_corr = []
    threshold = thresh

    print("\nColumn names:\n")

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                coll_corr.append(colname)
                print(colname)
    # print("Original Dataframe:")
    # print(dataframe)
    print("\nCorrelation Matrix:\n")
    corr_matrix = corr_matrix.dropna(how ='all')
    corr_matrix = corr_matrix.dropna(how ='all', axis = "columns")
    print(corr_matrix)

# calling correlation coefficient function
corr_co("previous_application.csv", df_previous_application_2, 0.85, False)
corr_co("current_application", df_application_data_2, 0.85, False)

# Pearson method correlation
# df_corr = df_previous_application_2.corr(method ='pearson')
# print(df_corr)

