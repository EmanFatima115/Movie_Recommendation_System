import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Load Data:
df=pd.read_csv('Mall_Customers.csv')
print(df.head())
print(df.shape)
# Cleaning Data:
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.columns)
# Features:
x=df[['Age','Annual Income (k$)','Spending Score (1-100)']]
# Scaling:
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
print(x_scaled[:5])
wcss=[]
# K_Model:
for i in range(1,11):
    model=KMeans(n_clusters=i,random_state=42,n_init='auto')
    model.fit(x_scaled)
    wcss.append(model.inertia_)

plt.plot(range(1,11),wcss,marker='o')
plt.xlabel('number of clusters')
plt.ylabel('Wcss')
plt.title('elbow method')
plt.show()
# Clustering:
model=KMeans(n_clusters=6,random_state=42,n_init='auto')
model.fit(x_scaled)
cluster_labels=model.labels_
print(cluster_labels)
df['cluster']=cluster_labels
print(df.head())
plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'],c=df['cluster'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
# Cluster profiling:
cluster_profile=df.groupby('cluster')[[
    'Age','Annual Income (k$)','Spending Score (1-100)'
]].mean()

df.to_csv(('customer_segments.csv'),index=False)
print(cluster_profile)
