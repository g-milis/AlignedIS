import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def cluster(embeddings, even_clusters=True, model="spiritlm"):
    def get_even_clusters(X, n_clusters):
        cluster_size = int(np.ceil(len(X) / n_clusters))
        kmeans = KMeans(n_clusters)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
        distance_matrix = cdist(X, centers)
        clusters = linear_sum_assignment(distance_matrix)[1] // cluster_size
        return clusters

    clusterings = {}
    for k in [2, 5, 8, 10, 13, 15, 17, 20, 40, 60, 80, 100]:
        cls_seeds = {}
        for i in range(50):
            print(k, i)
            if even_clusters:
                labels = get_even_clusters(embeddings, k)
            else:
                kmeans = KMeans(n_clusters=k, random_state=i)
                labels = kmeans.fit_predict(embeddings)
            cls_seeds[i] = labels
        clusterings[k] = cls_seeds


    if even_clusters:
        with open(f"embeddings/clusterings/{model}_even_clusterings.pkl", "wb") as f:
            pickle.dump(clusterings, f)
    else:
        with open(f"embeddings/clusterings/{model}_clusterings.pkl", "wb") as f:
            pickle.dump(clusterings, f)


def evaluate(embeddings, even_clusters=True, model="spiritlm"):
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    def compute_inertia(embeddings, labels, k):
        centroids = np.zeros((k, embeddings.shape[1]))
        for cluster_id in range(k):
            members = embeddings[labels == cluster_id]
            centroids[cluster_id] = members.mean(axis=0)
        distances = np.linalg.norm(embeddings - centroids[labels], axis=1)
        return np.sum(distances ** 2)

    if even_clusters:
        with open(f"embeddings/clusterings/{model}_even_clusterings.pkl", "rb") as f:
            clusterings = pickle.load(f)
    else:
        with open(f"embeddings/clusterings/{model}_clusterings.pkl", "rb") as f:
            clusterings = pickle.load(f)

    if even_clusters:
        ks = [2, 5, 8, 10, 13, 15, 17, 20, 40, 60, 80, 100]
    else:
        ks = [2, 5, 10, 15, 20, 40, 60, 80, 100]
    silhouette_means, silhouette_stds = [], []
    db_means, db_stds = [], []
    inertia_means, inertia_stds = [], []

    for k in ks:
        # try:
        #     clusterings[k]
        # except:
        #     ks.remove(k)
        #     continue

        sil_scores, db_scores, inertias = [], [], []
        for i in range(50):
            labels = np.array(clusterings[k][i])
            sil_scores.append(silhouette_score(embeddings, labels))
            db_scores.append(davies_bouldin_score(embeddings, labels))
            inertias.append(compute_inertia(embeddings, labels, k))
        silhouette_means.append(np.mean(sil_scores))
        silhouette_stds.append(np.std(sil_scores))
        db_means.append(np.mean(db_scores))
        db_stds.append(np.std(db_scores))
        inertia_means.append(np.mean(inertias))
        inertia_stds.append(np.std(inertias))

    # Plot all metrics
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Silhouette (left y-axis)
    ax1.errorbar(ks, silhouette_means, yerr=silhouette_stds, fmt='-o', color='tab:blue', label='Silhouette Score')
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Silhouette Score", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Davies-Bouldin (right y-axis)
    ax2 = ax1.twinx()
    ax2.errorbar(ks, db_means, yerr=db_stds, fmt='-s', color='tab:red', label='Davies-Bouldin Score')
    ax2.set_ylabel("Davies-Bouldin Score", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Inertia (additional axis)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.05))
    ax3.errorbar(ks, inertia_means, yerr=inertia_stds, fmt='-^', color='tab:green', label='Inertia')
    ax3.set_ylabel("Inertia", color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')

    # Legends
    lines, labels = [], []
    for ax in [ax1, ax2, ax3]:
        line, label = ax.get_legend_handles_labels()
        lines += line
        labels += label
    ax1.legend(lines, labels, loc='upper left')

    plt.title("Clustering Metrics vs Number of Clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/plots/clustering{'_even' if even_clusters else ''}_{model}_metrics_all.png")
    plt.close()


if __name__ == "__main__":

    model = "spiritlm"
    embeddings = torch.load(f"embeddings/{model}.pt")

    for model in ["spiritlm", "speechgpt"]:
        for even in [True, False]:
            embeddings = torch.load(f"embeddings/{model}.pt")

            cluster(embeddings, even_clusters=even, model=model)

            evaluate(embeddings, even_clusters=even, model=model)
