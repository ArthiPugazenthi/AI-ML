# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
!pip install kneed
from kneed import KneeLocator

# Load dataset
df = pd.read_csv("Customer Data.csv")

df.head()

df.info()

df.isnull().sum()

# Drop 'CUST_ID' as it is an identifier
df.drop(columns=['CUST_ID'], inplace=True)

#Handle missing values using median imputation
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

#Normalize the data using MinMaxScaler for better feature scaling
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

df_scaled.isnull().sum()

#Apply PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])

#Explained Variance Ratio
explained_variance = pca.explained_variance_ratio_
total_variance_explained = sum(explained_variance)
print(f"Explained Variance Ratio: {explained_variance}")
print(f"Total Variance Explained: {total_variance_explained:.4f}")

#Find the optimal number of clusters using the Elbow Method with KneeLocator
inertia = []
k_values = range(2, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_pca)
    inertia.append(kmeans.inertia_)

#Automatically determine the best k using KneeLocator
knee_locator = KneeLocator(k_values, inertia, curve="convex", direction="decreasing")
optimal_k = knee_locator.elbow
print(f"Optimal number of clusters: {optimal_k}")

#Plot the Elbow Method Graph
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--', color='b', label="Inertia")
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.legend()
plt.show()

#Apply K-Means with the optimal k
kmeans = KMeans(n_clusters=optimal_k)
df_pca['Cluster'] = kmeans.fit_predict(df_pca)

#Compute Silhouette Score
silhouette_avg = silhouette_score(df_pca.drop(columns=['Cluster']), df_pca['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Plot the clusters on a 2D PCA Map
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_pca.iloc[:, 0], y=df_pca.iloc[:, 1], hue=df_pca['Cluster'], palette='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()