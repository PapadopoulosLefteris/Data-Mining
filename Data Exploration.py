import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('healthcare-dataset-stroke-data.csv')

'''Κατηγορικά δεδομένα σε αριθμητικά'''
##df['gender']= df['gender'].astype('category').cat.codes
##df['ever_married'] = df['ever_married'].astype('category').cat.codes
##df['work_type'] = df['work_type'].astype('category').cat.codes
##df['Residence_type'] = df['Rσesidence_type'].astype('category').cat.codes
##df['smoking_status'] = df['smoking_status'].astype('category').cat.codes

'''Κατανομές'''                                                
##sns.displot(df,x=df['bmi'],hue='stroke',kind='kde')
                                                                      

'''Πίνακας συσχέτισης'''
##corr = df.corr(method = 'spearman')
##sns.heatmap(corr, annot=True)


'''Ιστόγραμμα'''
####sns.histplot(df['smoking_status'])
plt.show()

