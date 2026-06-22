# =========================
# UNSUPERVISED LEARNING - CUSTOMER SEGMENTATION
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# 1. LOAD OR CREATE DATASET (20+ FEATURES REQUIRED)
# =========================

try:
    df = pd.read_csv("customer_data.csv")
    print("Dataset loaded successfully")

except:
    print("No dataset found → generating retail-like dataset")

    np.random.seed(42)
    df = pd.DataFrame({
        "Income": np.random.randint(20000, 150000, 1200),
        "Age": np.random.randint(18, 70, 1200),
        "SpendingScore": np.random.randint(1, 100, 1200),
        "PurchaseFreq": np.random.randint(1, 60, 1200),
        "OnlineSpend": np.random.randint(100, 15000, 1200),
        "StoreSpend": np.random.randint(100, 15000, 1200),
        "DiscountUsage": np.random.randint(0, 40, 1200),
        "Visits": np.random.randint(1, 50, 1200),
        "MembershipYears": np.random.randint(0, 15, 1200),
        "Returns": np.random.randint(0, 15, 1200),

        # extra 10+ features to satisfy requirement
        "Electronics": np.random.randint(0, 2000, 1200),
        "Clothing": np.random.randint(0, 2000, 1200),
        "Grocery": np.random.randint(0, 2000, 1200),
        "Beauty": np.random.randint(0, 2000, 1200),
        "Sports": np.random.randint(0, 2000, 1200),
        "Travel": np.random.randint(0, 2000, 1200),
        "Books": np.random.randint(0, 2000, 1200),
        "Furniture": np.random.randint(0, 2000, 1200),
        "OnlineOrders": np.random.randint(0, 300, 1200),
        "StoreOrders": np.random.randint(0, 300, 1200),
    })

# =========================
# 2. PREPROCESSING
# =========================

X = df.select_dtypes(include=[np.number])
X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3. PCA (REDUCE TO 2 DIMENSIONS)
# =========================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# =========================
# 4. ELBOW METHOD (FIND OPTIMAL K)
# =========================

wcss = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(K, wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# =========================
# 5. SILHOUETTE SCORE
# =========================

sil_scores = []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    sil_scores.append(silhouette_score(X_pca, labels))

plt.figure()
plt.plot(K, sil_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.show()

best_k = K[np.argmax(sil_scores)]
print("Optimal number of clusters:", best_k)

# =========================
# 6. FINAL K-MEANS MODEL
# =========================

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

df["Cluster"] = clusters

# =========================
# 7. VISUALIZATION OF CLUSTERS
# =========================

plt.figure()
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="viridis")
plt.title("Customer Segmentation (PCA + KMeans)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# =========================
# 8. BUSINESS PERSONAS
# =========================

print("\n===== CUSTOMER PERSONAS =====")

for i in range(best_k):
    group = df[df["Cluster"] == i]

    print("\nCluster:", i)
    print("Size:", len(group))

    # interpret using averages
    print("Average Income:", group["Income"].mean())
    print("Average Spending Score:", group["SpendingScore"].mean())

    if group["Income"].mean() < 60000:
        print("Persona: Budget Customers")
    elif group["Income"].mean() < 100000:
        print("Persona: Mid-range Customers")
    else:
        print("Persona: Premium Customers")