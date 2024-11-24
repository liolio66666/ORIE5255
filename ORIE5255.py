import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("transactions_uniswap.csv")

df['value'] = pd.to_numeric(df['value'], errors='coerce')
df['value'] = df['value']/1e18
df['timeStamp'] = pd.to_datetime(df['timeStamp'],unit="s")


print(df.columns)
#print("Missing Values:\n", df.isnull().sum())
df = df[df["isError"]==0]

threshold_90 = df["value"].quantile(0.9)

plt.figure(figsize=(10, 6))
sns.histplot(df[df['value']<=threshold_90]['value'], bins=25, kde=True)
plt.title('Distribution of Transaction Values within 90% threshold')
plt.xlabel('Transaction Amount (ETH)')
plt.ylabel('Frequency')
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(df['gasUsed'], bins=25, kde=True)
plt.title('Distribution of Gas Used')
plt.xlabel('Gas Used')
plt.ylabel('Frequency')
plt.close()
################################################################
################################################################
################################################################


pca_data = df[["gas","gasPrice", "value", "gasUsed", "transactionIndex"]]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(pca_data)

pca = PCA(n_components=2)  
pca_data = pca.fit_transform(scaled_data)

print("PCA Result (First Two Principal Components):")
print(pca_data)

explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance Ratio:", explained_variance)

feature_names = ["gas", "gasPrice", "value", "gasUsed", "transactionIndex"]
components_df = pd.DataFrame(pca.components_, columns=feature_names, index=["PC1", "PC2"])
print("\nPCA Components (Loadings):")
print(components_df)



components_df.T.plot(kind="bar", figsize=(10, 6))
plt.title("Feature Contributions to Principal Components")
plt.xlabel("Features")
plt.ylabel("Contribution")
plt.xticks(rotation=0)
plt.legend(title="Principal Components")
plt.close()
################################################################
################################################################
################################################################


inertia = []
silhouette_scores = []
k_range = range(2, 10)  

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_data)
    inertia.append(kmeans.inertia_) 
    silhouette_scores.append(silhouette_score(pca_data, kmeans.labels_))

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')

plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
plt.title('Silhouette Scores for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(pca_data)
cluster_labels = kmeans.labels_

df['Cluster'] = cluster_labels
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue = cluster_labels, palette='Set2', alpha=0.8)
plt.title('Cluster Visualization in PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue= df['functionName'], palette='Set2', alpha=0.8)
plt.title('Cluster Visualization by functionName')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()



cluster_summary = df.groupby('Cluster')[["gas", "gasPrice", "value", "gasUsed", "transactionIndex"]].mean()
print("Cluster Summary Statistics:")
print(cluster_summary)

