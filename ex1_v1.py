import pandas as pd
import pandasql as ps
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main_menu():

    df = pd.read_csv("../healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv")
    while (True):
        #x = input("Ερώτημα Α (a) ή Β (b)\n")
        x="b"
        if x=="a":
            exerciseA(df)

        elif x=="b":
            #x = input("Ερώτηση 1, 2 ή 3\n")
            x="3"
            if x == "1": exerciseB_1(df)
            elif x=="2": exerciseB_2(df)
            elif x=="3": exerciseB_3(df); break
            else: continue

        else: break



def exerciseA(df):
    
    corr = df.corr(method = 'spearman')

    sns.heatmap(corr, annot=True)

    plt.show()


def exerciseB_1(df):
    print("Before: ", list(df.columns.values),"\n")

    df_1 = df.drop("bmi", axis="columns", inplace=False) #drop a column

    print("After: ", list(df_1.columns.values), "\n")


def exerciseB_2(df):
    
    mean_bmi = df['bmi'].mean()
    print("Mean of BMI: ", mean_bmi, "\n")

    df_2 = df.fillna({'bmi':mean_bmi}) #fills the NA values of the column "BMI" with the mean

    total_na_df = df['bmi'].isna().sum()
    total_na_df_2 =  df_2['bmi'].isna().sum()


    print("Total NaN values in column 'bmi':")
    print("---------------------------------")
    print("Before: {}\nAfter: {}\n".format(total_na_df,total_na_df_2))


def exerciseB_3(df):
    
    linearReg = LinearRegression()

    data = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']]

    data.fillna({'bmi':0}, inplace=True)

    X = data.drop(columns='bmi')
    y = data['bmi'].values
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    linearReg.fit(X_train, y_train)

    y_pred = linearReg.predict(X_test)

    print(len(y_pred))




if __name__ == "__main__":
    main_menu()
