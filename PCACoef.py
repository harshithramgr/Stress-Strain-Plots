# ---------------------------------------------------------------
# 0) Imports
# ---------------------------------------------------------------
import csv, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# 1) Load ragged coefficient rows
# ---------------------------------------------------------------
rows = []
with open("coefficients.csv", newline="") as f:
    for r in csv.reader(f):
        rows.append([float(x) for x in r if x])

# ---------------------------------------------------------------
# 2) Pad with NaN → rectangular array
# ---------------------------------------------------------------
max_len = max(map(len, rows))
X = pd.DataFrame([r + [np.nan]*(max_len-len(r)) for r in rows]).values
n_samples = X.shape[0]

# ---------------------------------------------------------------
# 3) Impute missing → standardise
# ---------------------------------------------------------------
X_std = StandardScaler().fit_transform(SimpleImputer(strategy="mean").fit_transform(X))

# ---------------------------------------------------------------
# 4) Pick k via elbow & silhouette
# ---------------------------------------------------------------
k_min, k_max = 2, min(10, n_samples-1)
inertias, sils = [], []

for k in range(k_min, k_max+1):
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    labels_tmp = km.fit_predict(X_std)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_std, labels_tmp))

best_k = range(k_min, k_max+1)[np.argmax(sils)]
print(f"\nChosen k = {best_k}  (highest silhouette)")

# quick elbow/silhouette plot (optional)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(range(k_min, k_max+1), inertias, marker='o'); ax1.set_title('Elbow'); ax1.set_xlabel('k')
ax2.plot(range(k_min, k_max+1), sils, marker='o');     ax2.set_title('Silhouette'); ax2.set_xlabel('k')
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------
# 5) Final K-means
# ---------------------------------------------------------------
best_k = 10
kmeans  = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
labels  = kmeans.fit_predict(X_std)
assign  = pd.DataFrame({"curve_index": np.arange(1, n_samples+1), "cluster": labels})

print("\nCurve ↔ Cluster mapping:")
print(assign.to_string(index=False))

print("\nCounts per cluster:")
for cl, cnt in assign['cluster'].value_counts().sort_index().items():
    print(f"  Cluster {cl}: {cnt} curves")

print("\nMembers of each cluster:")
for cl in sorted(assign['cluster'].unique()):
    print(f"  Cluster {cl}: {assign.loc[assign['cluster']==cl,'curve_index'].tolist()}")

# ---------------------------------------------------------------
# 6) 2-D projections (PCA + safe t-SNE)
# ---------------------------------------------------------------
# ---- PCA for everybody
pca2 = PCA(n_components=2, random_state=0).fit(X_std)
pc   = pca2.transform(X_std)

# ---- t-SNE for everybody
if n_samples >= 3:
    perp_data = max(2, min(30, n_samples-1, n_samples//3))
    print(f"\n[t-SNE] Data perplexity = {perp_data}")
    tsne_data = TSNE(n_components=2, perplexity=perp_data,
                     learning_rate="auto", init="pca",
                     random_state=0).fit_transform(X_std)
else:
    tsne_data = None
    print("\n[t-SNE] Skipped (n_samples < 3)")

# ---- centroids
centroids = kmeans.cluster_centers_
cent_pc   = pca2.transform(centroids)

if tsne_data is not None and best_k >= 3:
    perp_cent = max(2, min(30, best_k-1, best_k//2))
    print(f"[t-SNE] Centroid perplexity = {perp_cent}")
    tsne_cent = TSNE(n_components=2, perplexity=perp_cent,
                     learning_rate="auto", init="random",
                     random_state=1).fit_transform(centroids)
else:
    tsne_cent = None

# ---------------------------------------------------------------
# 7) Helper for scatter plots
# ---------------------------------------------------------------
def scatter(ax, emb, cent, title):
    sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=labels,
                    palette='tab10', s=80, alpha=.85, ax=ax, legend=False)
    if cent is not None:
        ax.scatter(cent[:,0], cent[:,1], marker='X', s=160,
                   edgecolor='k', linewidth=1.2, zorder=10)
    ax.set_title(title); ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')

# ---------------------------------------------------------------
# 8) Show PCA and t-SNE plots
# ---------------------------------------------------------------
n_plots = 2 if tsne_data is not None else 1
fig, axs = plt.subplots(1, n_plots, figsize=(12, 5))

scatter(axs if n_plots==1 else axs[0],
        pc, cent_pc,
        f"PCA  (var = {pca2.explained_variance_ratio_[:2].sum()*100:.1f} %)")

if tsne_data is not None:
    scatter(axs[1], tsne_data, tsne_cent, "t-SNE")

plt.suptitle("B-spline Coefficient Space – Cluster Maps", y=1.02, fontsize=14)
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------
# 9) Save assignments
# ---------------------------------------------------------------
assign.to_csv("cluster_assignments.csv", index=False)
print("\nCluster labels saved → cluster_assignments.csv")
