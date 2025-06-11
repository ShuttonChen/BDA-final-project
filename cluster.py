#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def cluster_and_save_kmeans(input_csv: str, output_csv: str):
    """
    讀取 input_csv（格式為 id,dim1,dim2,…,dimN），
    自動計算 n=維度數，群數 k=max(2, 4*n-1)，
    進行標準化、PCA（保留 75% 變異）、
    接著使用 KMeans++ 分群，
    並輸出 id + label 到 output_csv。
    同時印出分群品質指標。
    """
    # 讀取資料並檢查 id 欄
    df = pd.read_csv(input_csv)
    if 'id' not in df.columns:
        raise ValueError(f"{input_csv} 中找不到 'id' 欄位")
    ids = df['id']
    features = df.drop(columns=['id']).values

    # 計算群數
    n_dims = features.shape[1]
    k = max(2, 4 * n_dims - 1)
    print(f"檔案 {input_csv}：原始維度 = {n_dims}, 分群數 k = {k}")

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # PCA 降噪：保留 75% 變異
    pca = PCA(n_components=0.75, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    print(f"PCA 降維：從 {n_dims} 維 → {X_reduced.shape[1]} 維")

    # KMeans++ 分群
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        random_state=42
    )
    labels = kmeans.fit_predict(X_reduced)

    # 分群評估
    sil_score = silhouette_score(X_reduced, labels)
    db_score = davies_bouldin_score(X_reduced, labels)
    print(f"Silhouette Score = {sil_score:.4f}")
    print(f"Davies–Bouldin Score = {db_score:.4f}")

    # 儲存結果
    submission = pd.DataFrame({'id': ids, 'label': labels})
    submission.to_csv(output_csv, index=False)
    print(f"已輸出 {output_csv}\n")


if __name__ == "__main__":
    # 分別對公開與私有資料集執行
    cluster_and_save_kmeans("public_data.csv",  "public_submission.csv")
    cluster_and_save_kmeans("private_data.csv", "private_submission.csv")
