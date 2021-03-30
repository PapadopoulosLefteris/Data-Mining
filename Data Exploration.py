import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('healthcare-dataset-stroke-data.csv')

'''Γενικές πληροφορίες του dataframe'''
print(df.head())
print(df.info())

'''Ιστόγραμμα'''
for i in df.columns.values:
    if i!='id' and i!='stroke':   
        histplot_fig = sns.histplot(df[i])
        histplot_fig.figure.savefig('Histogram_'+i+'.png')
        plt.clf()
        

'''Κατηγορικά δεδομένα σε αριθμητικά'''
df['gender']= df['gender'].astype('category').cat.codes
df['ever_married'] = df['ever_married'].astype('category').cat.codes
df['work_type'] = df['work_type'].astype('category').cat.codes
df['Residence_type'] = df['Residence_type'].astype('category').cat.codes
df['smoking_status'] = df['smoking_status'].astype('category').cat.codes



'''Κατανομές'''
for i in df.columns.values:
    if i!=df.columns.values[0] and i!=df.columns.values[-1]:
        print(i)
        distplot_fig = sns.displot(df,x=df[i],hue='stroke',kind='kde')
        distplot_fig.savefig('Distribution_'+i+'.png')                                     

plt.clf()
'''Πίνακας συσχέτισης'''
corr = df.corr(method = 'spearman')
heatmap_fig = sns.heatmap(corr, annot=True)
heatmap_fig.figure.savefig('Heatmap Correlation')



