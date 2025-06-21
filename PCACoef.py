import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------------------------------------------------------
# 1) Load ragged coefficient rows
# ---------------------------------------------------------------
rows = []
with open("coefficients.csv", newline="") as f:
    for r in csv.reader(f):
        rows.append([float(x) for x in r if x != ""])

# ---------------------------------------------------------------
# 2) Pad with NaN → rectangular array
# ---------------------------------------------------------------
max_len = max(len(r) for r in rows)
padded  = [r + [np.nan]*(max_len - len(r)) for r in rows]
X = pd.DataFrame(padded).values            # (n_samples × max_len)

# ---------------------------------------------------------------
# 3) Impute missing → standardise
# ---------------------------------------------------------------
X_imp = SimpleImputer(strategy="mean").fit_transform(X)
X_std = StandardScaler().fit_transform(X_imp)

# ---------------------------------------------------------------
# 4) ELBOW PLOT to help pick k
# ---------------------------------------------------------------
max_k = 10            # check k = 1 … 10
inertias = []

for k in range(1, max_k + 1):
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    km.fit(X_std)
    inertias.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, max_k + 1), inertias, marker="o")
plt.xticks(range(1, max_k + 1))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster SSE)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 5) Choose k (here fixed at 3) and run K-means
# ---------------------------------------------------------------
k = 3
kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
labels = kmeans.fit_predict(X_std)

print("\nCluster sizes:")
for lab, cnt in zip(*np.unique(labels, return_counts=True)):
    print(f"  Cluster {lab} → {cnt} curves")

# ---------------------------------------------------------------
# 6) 2-D PCA projection for visualisation
# ---------------------------------------------------------------
pca = PCA(n_components=2, random_state=0).fit(X_std)
pc  = pca.transform(X_std)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(pc[:, 0], pc[:, 1], c=labels, cmap="Set1", s=80, alpha=0.8)

# label each point by curve index (1-based)
for i, (x, y) in enumerate(pc, start=1):
    plt.text(x, y, str(i), fontsize=8, ha="left", va="center")

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
plt.title("B-spline Coefficient Space – K-means Clusters")
plt.grid(True)
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 7) Save cluster assignments (optional)
# ---------------------------------------------------------------
pd.DataFrame(
    {"curve_index": np.arange(1, len(labels) + 1), "cluster": labels}
).to_csv("cluster_assignments.csv", index=False)

print("\nCluster labels saved ➜  cluster_assignments.csv")
