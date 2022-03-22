import pandas as pd
import numpy as np

df = pd.read_csv(".\datasets\application_data.csv")
df.shape



if __name__ == "__main__":
    print ("Executed when invoked directly")
else:
    print ("Executed when imported")