import pandas as pd
import warnings

df = pd.read_csv("../healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv")
warnings.filterwarnings("ignore") #ignores pandas' warnings


print(df.isna().sum())