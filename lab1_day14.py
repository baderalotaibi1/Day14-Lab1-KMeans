
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report



df=pd.read_csv('C:/Users/bader/OneDrive/gitlesson/Day14-Lab1-KMeans/College_Data')
print(df.head())

sns.scatterplot(x='Room.Board',y='Grad.Rate',data=df,hue=df['Private'])
plt.title('scatterplot of Grad.Rate versus Room.Board')
#plt.show()

sns.scatterplot(x ='Outstate', y ='F.Undergrad', data = df,  hue = 'Private', alpha = 0.3)
plt.title('scatterplot of F.Undergrad versus Outstate')
#plt.show()

fact= sns.FacetGrid(df, hue = 'Private')
fact.map(plt.hist, 'Outstate', bins = 20, alpha = 0.3)
plt.title('FacetGrid showing Out of State ')
#plt.show()

fact= sns.FacetGrid(df, hue = 'Private')
fact.map(plt.hist, 'Grad.Rate', bins = 20, alpha = 0.3)
plt.title('FacetGrid showing Grad.Rate')
#plt.show()
#Notice how there seems to be a private school with a graduation rate of higher than 100%.What is the name of that school?
print(df[df['Grad.Rate']>100])
#Set that school's graduation rate to 100 then re-do the histogram visualization.
schools=df.loc[df['Grad.Rate'] > 100, 'Grad.Rate'] = 100
print('#Set that schools graduation rate to 100 then re-do the histogram visualization.\n',schools)
fact= sns.FacetGrid(df, hue = 'Private')
fact.map(plt.hist, 'Grad.Rate', bins = 20, alpha = 0.3)
plt.title('FacetGrid showing Grad.Rate')
#plt.show()

#Fit the model to all the data except for the Private label.

kmeans = KMeans(n_clusters=2)
x=df.drop('Private', axis = 1)
x= x.iloc[: , 1:]
print(x)
kmeans.fit(x)

#What are the cluster center vectors?

print(kmeans.cluster_centers_)

#Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.

df['Cluster'] = df['Private'].apply(lambda x: 1 if x == 'Yes' else 0)
print(df.head())
#Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.


print(confusion_matrix(df['Cluster'], kmeans.labels_),'\n')
print(classification_report(df['Cluster'], kmeans.labels_))

