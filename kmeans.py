import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df=pd.read_csv('data/clean/orders.csv')

completed=df[df['status']=='completed']
completed['order_ts']=pd.to_datetime(completed['order_ts'])

snapshot=completed['order_ts'].max()  # reference date for recency

rfm = completed.groupby('customer_id').agg(
    Recency=('order_ts', lambda x: (snapshot - x.max()).days),
    Frequency=('order_id', 'nunique'),
    Monetary=('order_amount', 'sum')
).reset_index()

print(df.head(),"\n")
print(rfm.shape[0])
print("Receny median:", rfm["Recency"].median())
print("minimum recency:", rfm["Recency"].min())
print("maximum recency:",rfm["Recency"].max())
print("Frequency mean:",rfm["Frequency"].mean().round(2))
print("maximum frequency:",rfm["Frequency"].max())
print("Monetary mean:",rfm["Monetary"].mean().round(2))
print("maximum monetary",rfm["Monetary"].max().round(2))

features=["Recency","Frequency","Monetary"]
X=rfm[features].copy()
print(X.isnull().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
silhouettes=[]

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,   
        random_state=42, 
        n_init=10         # Run multiple initializations and keep the best result
    )

    kmeans.fit(X_scaled)
    wcss.append(round(kmeans.inertia_,4)) 
    kmeans_silhouette = silhouette_score(
        X_scaled,
        kmeans.labels_
    )
    silhouettes.append(round(kmeans_silhouette,5))

plt.plot(range(2, 9), wcss, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS / Inertia")
plt.title("Elbow Method")
plt.show()

print("Inertia for each try:",wcss)
print("K-Means Silhouette Score for each try:",silhouettes)

#taking the value of K as 4 based on model analysis
k=4
km = KMeans(n_clusters=k,n_init=10,random_state=42)
rfm['segment_id'] = km.fit_predict(X_scaled)

segment_summary = rfm.groupby('segment_id').agg(
    size=('customer_id', 'count'),
    mean_recency=('Recency', 'mean'),
    mean_frequency=('Frequency', 'mean'),
    mean_monetary=('Monetary', 'mean')
).round(2)

print(segment_summary)

pca=PCA(n_components=2)
Xpca=pca.fit_transform(X_scaled)

print("Variance per component:", pca.explained_variance_ratio_.round(3))

rfm['is_premium']=rfm['Monetary']>=4077.25  #creating threshold for premium customers

name_map = {
    0: 'Frequent Buyer',
    1: 'Premium Buyer',
    2: 'At Risk Buyer',
    3: 'Occasional Buyer'
}
rfm['segment_name'] = rfm['segment_id'].map(name_map)

rfm.to_csv('outputs/customer_segments.csv',index=False)