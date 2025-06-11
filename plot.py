#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

def visualize(dataset_name, data_path, submission_path):
    """
    做 PCA 降維並繪製：
      1) 未分群的 PCA 投影圖
      2) 已分群的 PCA 投影圖
    並將圖存成四張檔案。
    """
    # 讀資料並合併分群結果
    df_data = pd.read_csv(data_path)              # id, dim1, dim2, …
    df_sub  = pd.read_csv(submission_path)        # id, label
    df      = df_data.merge(df_sub, on='id')

    # 特徵矩陣與標籤
    X      = df.drop(columns=['id', 'label']).values
    labels = df['label'].values

    # 標準化 + PCA(保留75%變異)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.75, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    # 2D 還是 3D？
    dims = X_pca.shape[1]

    # 1) PCA 投影 (不著色)
    if dims == 2:
        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(X_pca[:,0], X_pca[:,1], s=15, alpha=0.8)
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var)")
        ax.set_title(f"{dataset_name} dataset PCA")
        plt.tight_layout()
        fig.savefig(f"{dataset_name.lower()}_pca.png")
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], s=15, alpha=0.8)
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var)")
        ax.set_zlabel(f"PC3 ({explained[2]:.1f}% var)")
        ax.set_title(f"{dataset_name} dataset PCA (3D)")
        plt.tight_layout()
        fig.savefig(f"{dataset_name.lower()}_pca_3d.png")
        plt.close(fig)

    # 2) PCA 投影 (著色)
    if dims == 2:
        fig, ax = plt.subplots(figsize=(6,5))
        scatter = ax.scatter(
            X_pca[:,0], X_pca[:,1],
            c=labels, cmap='tab20', s=15, alpha=0.8
        )
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var)")
        ax.set_title(f"{dataset_name} dataset PCA + Clusters")
        cbar = plt.colorbar(scatter, ax=ax, ticks=sorted(set(labels)))
        cbar.set_label('Cluster Label')
        plt.tight_layout()
        fig.savefig(f"{dataset_name.lower()}_pca_cluster.png")
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            X_pca[:,0], X_pca[:,1], X_pca[:,2],
            c=labels, cmap='tab20', s=15, alpha=0.8
        )
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var)")
        ax.set_zlabel(f"PC3 ({explained[2]:.1f}% var)")
        ax.set_title(f"{dataset_name} dataset PCA + Clusters (3D)")
        cbar = fig.colorbar(scatter, ax=ax, ticks=sorted(set(labels)), pad=0.1)
        cbar.set_label('Cluster Label')
        plt.tight_layout()
        fig.savefig(f"{dataset_name.lower()}_pca_3d_cluster.png")
        plt.close(fig)

if __name__ == "__main__":
    # 公開資料集
    visualize(
        dataset_name="Public",
        data_path="/home/b11902157/big_data/public_data.csv",
        submission_path="/home/b11902157/big_data/public_submission.csv"
    )
    # 私有資料集
    visualize(
        dataset_name="Private",
        data_path="/home/b11902157/big_data/private_data.csv",
        submission_path="/home/b11902157/big_data/private_submission.csv"
    )
