import pandas as pd
#import pandasql as ps
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn import metrics


def main_menu():

    df = pd.read_csv("../healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv")
    warnings.filterwarnings("ignore") #ignores pandas' warnings

    catCodes(df)

    while (True):
        x = input("Ερώτημα Α (a) ή Β (b) ή Γ (c)\n")
        if x=="a":
            exerciseA(df)

        elif x=="b":
            x = input("Ερώτηση 1, 2 ή 3\n")
            if x == "1": exerciseB_1(df)
            elif x=="2": exerciseB_2(df)
            elif x=="3": exerciseB_3(df)
            else: continue

        elif x == 'c':
            exerciseC(df)

        else: break



def exerciseA(df):
    '''
    Συνάρτηση που υλοποιεί την λύση της άσκησης A για το ερώτημα 1\r
    ___________\n
    Παράμετροι:\r
    df: DataFrame
    '''
    corr = df.corr(method = 'spearman')

    sns.heatmap(corr, annot=True)

    plt.show()


def exerciseB_1(df, exC_flag = False):
    '''
    Συνάρτηση που υλοποιεί την λύση της άσκησης Β1 για το ερώτημα 1\r
    ___________\n
    Παράμετροι:\r
    df: DataFrame\r
    [exC_flag = False]: boolean - Αν είναι true, η συνάρτηση επιστρέφει το νέο dataframe (χρήσιμο για την άσκηση Γ)
    '''
    if (not exC_flag): print("Before: ", list(df.columns.values),"\n")

    df_1 = df.drop(["bmi", "smoking_status"], axis="columns", inplace=False) #drop a column

    if (exC_flag): return df_1

    print("After: ", list(df_1.columns.values), "\n")


def exerciseB_2(df, exC_flag = False):
    '''
    Συνάρτηση που υλοποιεί την λύση της άσκησης Β2 για το ερώτημα 1\r
    ___________\n
    Παράμετροι:\r
    df: DataFrame\r
    [exC_flag = False]: boolean - Αν είναι true, η συνάρτηση επιστρέφει το νέο dataframe (χρήσιμο για την άσκηση Γ)
    '''
    mean_bmi = df['bmi'].mean()

    # print(df.groupby('smoking_status').size())

    #x = df['smoking_status'].count()

    #print(x)

    if (not exC_flag): print("Mean of BMI: ", mean_bmi, "\n")

    df_2 = df.fillna({'bmi':mean_bmi}) #fills the NA values of the column "BMI" with the mean

    total_na_df = df['bmi'].isna().sum()
    total_na_df_2 =  df_2['bmi'].isna().sum()

    if (exC_flag): return df_2

    print("Total NaN values in column 'bmi':")
    print("---------------------------------")
    print("Before: {}\nAfter: {}\n".format(total_na_df,total_na_df_2))


def exerciseB_3(df, exC_flag = False):
    '''
    Συνάρτηση που υλοποιεί την λύση της άσκησης Β3 για το ερώτημα 1\r
    ___________\n
    Παράμετροι:\r
    df: DataFrame\r
    [exC_flag = False]: boolean - Αν είναι true, η συνάρτηση επιστρέφει το νέο dataframe (χρήσιμο για την άσκηση Γ)
    '''

    '''
    linearReg = LinearRegression()

    data = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']]

    data.fillna({'bmi':0}, inplace=True)

    X = data.drop(columns='bmi')
    y = data['bmi'].values
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    linearReg.fit(X_train, y_train)

    y_pred = linearReg.predict(X_test)

    print(len(y_pred))
    '''

    linearReg = LinearRegression()

    data = df.copy(deep=True)

    Train_Features = data.dropna().drop(columns=['bmi', 'id', 'stroke'])
    Test_Labels = data.dropna()['bmi']

    X_train, X_test, y_train, y_test = train_test_split(Train_Features, Test_Labels, test_size=0.10)

    linearReg.fit(X_train, y_train)

    Test_Features = data[data['bmi'].isna()].drop(columns=['bmi', 'id', 'stroke'])

    predictions = linearReg.predict(Test_Features)

    if (not exC_flag): 
        x = input("Να εκτυπωθούν τα προβλεπόμενα αποτελέσματα; Y/n\n")
        if (x == 'y' or x=='Y'): print(predictions)

    data.loc[data.bmi.isna(), 'bmi'] = predictions

    before = df['bmi'].isna().sum()
    after = data['bmi'].isna().sum()

    if (exC_flag): return data

    y_pred = linearReg.predict(X_test)
    aError = y_test - y_pred
    plt.hist(aError, bins=10)
    plt.show()

    print("Total NaN values in column 'bmi':")
    print("---------------------------------")
    print("Before: {}\nAfter: {}\n".format(before,after))


def exerciseC(df):
    data_b1 = exerciseB_1(df, True)
    data_b2 = exerciseB_2(df, True)
    data_b3 = exerciseB_3(df, True)

    set_of_data = [data_b1, data_b2, data_b3]

    #print(set_of_data[0])
    #print(set_of_data[1])
    #print(set_of_data[2])

    
    i = 1

    for dt in set_of_data:
        X = dt.drop(columns=["stroke", "id"])
        y = dt["stroke"].values

        #print(dt)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)#, random_state=42, stratify=y)   #<-------     

        random_forest = RandomForestClassifier(n_estimators=100)
        
        #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        random_forest.fit(X_train, y_train)

        #Predictions
        y_pred = random_forest.predict(X_test)

        #Metrics
        f1 = metrics.f1_score(y_test, y_pred)

        pr = metrics.precision_score(y_test, y_pred)

        rec = metrics.recall_score(y_test, y_pred)

        #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Με δεδομένα τo μητρώο του ερωτήματος Β{}, η απόδοση του μοντέλου είναι:\nf1 score = {:f}\nprecision = {:f}\nrecall = {:f}\n\n\n".format(i,f1,pr,rec))
        i+=1

        '''https://stackoverflow.com/questions/62672842/how-to-improve-f1-score-for-classification'''

    
def catCodes(df):
    '''
    Μετατροπή των μη αλφαριθμητικών δεδομένων σε αλφαριθμητικά\r
    ___________\n
    Παράμετροι:\r
    df: DataFrame
    '''
    df['gender']= df['gender'].astype('category').cat.codes
    df['ever_married'] = df['ever_married'].astype('category').cat.codes
    df['work_type'] = df['work_type'].astype('category').cat.codes
    df['Residence_type'] = df['Residence_type'].astype('category').cat.codes

    df["smoking_status"].replace("Unknown", np.nan, inplace=True)

    df['smoking_status'] = df['smoking_status'].astype('category').cat.codes

if __name__ == "__main__":
    main_menu()
