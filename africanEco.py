#IMPORT PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#LOAD DATA AND OVERVIEW
df = pd.read_csv('Datasets/african_crises.csv')
df.describe()
df.info()
df.tail()
df.head()

#DATA VISUALIZATION
def EachFeatureHistogram(df):
	columns = df.columns.tolist()

	for column in columns:
		plt.figure(figsize=(8,8))
		df[column].hist(label= column)
		plt.legend()


def PlotEachFeature(df,x,y):
    unique_countries = df['country'].unique()
    
    for ucountry in unique_countries:
        plt.figure(figsize=(20,5))
        DfCountry = df[df['country']==ucountry]
        plt.grid()
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(np.arange(min(df[x]),max(df[x])+1,2),rotation='vertical')
        plt.plot(DfCountry[x],DfCountry[y],label = ucountry,marker='o')
        plt.plot([np.min(DfCountry[np.logical_and(DfCountry.country==ucountry,DfCountry.independence==1)][x]),np.min(DfCountry[np.logical_and(DfCountry.country==ucountry,DfCountry.independence==1)][x])],[np.min(DfCountry[DfCountry.country==ucountry][y]),np.max(DfCountry[DfCountry.country==ucountry][y])],color='red',linestyle='dashed',alpha=0.8,label="Independance")
        plt.legend()
        

def countPlotGroupByCountry(df,y,columns):
    for column in columns:
        plt.figure(figsize=(10,10))
        sns.countplot(y=df[y],hue=df[column])
        plt.legend(loc=0)
        plt.title(column)
        
def heatMapShow(df):
	sns.heatmap(df.corr())

        
columns = ["sovereign_external_debt_default","currency_crises","inflation_crises","banking_crisis","systemic_crisis"]
countPlotGroupByCountry(df,"country",columns)
PlotEachFeature(df,"year","inflation_annual_cpi")
EachFeatureHistogram(df)
heatMapShow(df)


