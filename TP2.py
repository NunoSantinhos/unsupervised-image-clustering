# -*- coding: utf-8 -*-
import os
import math
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, SelectKBest, chi2
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn import decomposition
from sklearn.manifold import TSNE, Isomap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.cluster import adjusted_rand_score
import tp2_aux as aux


os.environ["OMP_NUM_THREADS"] = '1'
os.environ['MPLBACKEND'] = 'TkAgg'


# ANOVA
def ANOVA(X, y, n_feats):
    f, prob = f_classif(X[y[:] > 0], y[y[:] > 0])
    lprob = prob.tolist()
    lprobc = lprob.copy()
    lprobc.sort()
    res = []
    for i in range(0, n_feats):
        res.append(lprob.index(lprobc[i]))
    return res

# K-means
def k_Means1(NClusters, columns, ids):
    kmeans = KMeans(n_clusters=NClusters, n_init="auto").fit(columns)
    labels = kmeans.predict(columns)
    aux.report_clusters(ids, labels, "./k_means/K_means_k" + str(NClusters) +
                        "_nf" + str(columns.shape[1]) + ".html")
    return labels

# PCA
def PCA(matrix):
    pca = decomposition.PCA(n_components=6)
    t_data = pca.fit_transform(matrix)
    return t_data

# t_SNE
def t_SNE(matrix):
    tsne = TSNE(n_components=6, method='exact')
    t_data = tsne.fit_transform(matrix)
    return t_data

# Isomap
def isomap(matrix):
    isomap = Isomap(n_components=6)
    t_data = isomap.fit_transform(matrix)
    return t_data


def external_index_calculation(KS, N):
    pairs = N * (N - 1) / 2
    total_positives = 0
    for i in range(0, KS.shape[0]):
        els = 0
        for j in range(0, KS.shape[1]):
            els += KS[i][j]
        total_positives += math.comb(int(els), 2)

    true_positives = 0
    for i in range(KS.shape[0]):
        for j in range(KS.shape[1]):
            if (KS[i][j] > 1):
                true_positives += math.comb(int(KS[i][j]), 2)

    false_positives = total_positives - true_positives
    total_negatives = pairs - total_positives
    false_negatives = 0

    for i in range(0, KS.shape[1]):
        for j in range(0, KS.shape[0]):
            match = KS[j][i]
            mismatches = 0
            for p in range(j + 1, KS.shape[0]):
                mismatches += KS[p][i]
            false_negatives += match * mismatches

    true_negatives = total_negatives - false_negatives
    assert (true_positives + false_positives + true_negatives + false_negatives == pairs)

    precision = true_positives / (false_positives + true_positives)
    recall = true_positives / (false_negatives + true_positives)
    f1 = 2 * ((precision * recall) / (precision + recall))
    rand = (true_positives + true_negatives) / pairs

    return precision, recall, f1, rand


def external_index(predicted_labels, labels, nClusters):
    predicted_labels = predicted_labels[labels[:] > 0]
    known_labels = labels[labels[:] > 0]

    KS = np.zeros((nClusters, 3))

    for i in range(0, predicted_labels.shape[0]):
        KS[predicted_labels[i]][known_labels[i] - 1] += 1

    prec_s, recall_s, f1_s, rand_index = external_index_calculation(KS, predicted_labels.shape[0])

    adj_rand_index = adjusted_rand_score(known_labels, predicted_labels)
    print("Precision Score:", prec_s)
    print("Recall Score:", recall_s)
    print("F1 Score:", f1_s)
    print("Rand Index Score:", rand_index)
    print("Adjusted Rand Index Score:", adj_rand_index)
    return prec_s, recall_s, f1_s, rand_index, adj_rand_index

# DBSCAN
def dbscan(feats, labels, epsilon):
    clustering = DBSCAN(eps=epsilon).fit_predict(feats)
    aux.report_clusters(labels[:, 0], clustering, "./dbscan/epsilon_" + str(epsilon) +
                        "_nf" + str(feats.shape[1]) + ".html")
    return clustering

# Affinity
def affinity(feats, labels, damping):
    aff_prop = AffinityPropagation(damping=damping).fit(feats)
    labels = aff_prop.labels_
    silhouette = silhouette_score(feats, labels)
    adj_rand_index = adjusted_rand_score(labels, labels)  # Calculando apenas para evitar erro
    n_clusters = len(np.unique(labels))
    aux.report_clusters(labels, aff_prop, "./affinity/damping_" + str(damping) +
                        "_nf" + str(feats.shape[1]) + ".html")
    return labels, silhouette, adj_rand_index, n_clusters



def Main():
    lines = open("labels.txt").readlines()
    labels = []
    for line in lines:
        line = line.strip('\n')
        parts = line.split(',')
        labels.append((int(parts[0]), int(parts[1])))
    labels = np.array(labels)
    label_images = aux.images_as_matrix()


    pca_features = PCA(label_images)
    tsne_features = t_SNE(label_images)
    isomap_features = isomap(label_images)
    all_features = np.concatenate((pca_features, tsne_features, isomap_features), axis=1)

    nomes_colunas = ['i', 'n_clusters', 'Precision Score', 'Recall Score', 'F1 Score', 'Rand Index Score',
                     'Adjusted Rand Index Score', 'Silhouette Score']
    nomes_colunas2 = ['i', 'epsi', 'Precision Score', 'Recall Score', 'F1 Score', 'Rand Index Score',
                     'Adjusted Rand Index Score', 'Silhouette Score']
    df_k = pd.DataFrame(columns=nomes_colunas)
    df_db = pd.DataFrame(columns=nomes_colunas2)
    silhouette_scores = []
    silhouette_scores_db = []

    # K-means
    for i in range(1, 8):
        best_features = ANOVA(all_features, labels[:, 1], i)
        array_best_features = np.array(all_features[:, best_features])

        for k in range(2, 11):
            kmeans = k_Means1(k, array_best_features, labels[:, 0])
            e = external_index(kmeans, labels[:, 1], k)
            silhouette_s = silhouette_score(array_best_features, kmeans)
            silhouette_scores.append(silhouette_s)

            new_row = pd.DataFrame([[i, k, e[0], e[1], e[2], e[3], e[4], silhouette_s]], columns=nomes_colunas)
            df_k = pd.concat([df_k, new_row], ignore_index=True)

    # DBSCAN
    epsi = [45, 440, 900, 1000, 1000, 1050, 1050, 1050]
    for i in range(1, 8):
        db = dbscan(array_best_features, labels, epsi[i - 1])
        unique_clusters = np.unique(db[db != -1])
        if unique_clusters.size < 2:
            adjusted_rand_s = adjusted_rand_score(labels[:, 1], db)
            new_row_db = pd.DataFrame([[i, epsi[i - 1], 0, 0, 0, 0, 0, 0]], columns=nomes_colunas2)
            df_db = pd.concat([df_db, new_row_db], ignore_index=True)
            print("um cluster para epsi =", epsi[i - 1])
            continue

        ex = external_index(db, labels[:, 1], max(db) + 1)
        silhouette_s = silhouette_score(array_best_features, db)
        silhouette_scores_db.append(silhouette_s)

        new_row_db = pd.DataFrame([[i, epsi[i - 1], ex[0], ex[1], ex[2], ex[3], ex[4], silhouette_s]],
                                  columns=nomes_colunas2)
        df_db = pd.concat([df_db, new_row_db], ignore_index=True)


    labels_aff, silhouette_aff, adj_rand_index_aff, n_clusters_aff = affinity(array_best_features, labels[:, 1], 0.50)

    print("Silhouette Score (Affinity Propagation):", silhouette_aff)
    new_row_aff = pd.DataFrame([["AffinityPropagation", len(np.unique(labels_aff)), 0, 0, 0, 0, adj_rand_index_aff,
                                  silhouette_aff]], columns=nomes_colunas)
    if df_k.empty:
        df_k = new_row_aff.copy()
    else:
        df_k = pd.concat([df_k, new_row_aff], ignore_index=True)

    #print(df_k, df_db)
    print(df_k['Precision Score'], df_db['Precision Score'])


    # Gráfico 1: n_clusters vs Silhouette Score
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_k, x='n_clusters', y='Silhouette Score', marker='o')
    plt.title('Silhouette Score vs Number of Clusters : k_means')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

    # Gráfico 2: n_clusters vs Silhouette Score com cores de i
    df_k_filtered = df_k[df_k['i'] != "AffinityPropagation"]

    # Plot the graph
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_k_filtered, x='n_clusters', y='Silhouette Score', hue='i', marker='o')
    plt.title('Silhouette Score vs N_clusters (colored by i): k_means')
    plt.xlabel('N_clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

    # Gráfico 3: n_clusters vs adjusted_rand
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_k, x='n_clusters', y='Adjusted Rand Index Score', marker='o')  ####pq é que nao pode ser y
    plt.title('Silhouette Score vs adjusted_rand: k_means')
    plt.xlabel('Number of Clusters')
    plt.ylabel('adjusted_rand_s')
    plt.grid(True)
    plt.show()

    # Gráfico 4: epsi vs Silhouette Score
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_db, x='epsi', y='Silhouette Score', marker='o')
    plt.title('Epsi vs Silhouette Score: db')
    plt.xlabel('Epsi')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

    # Gráfico 5: epsi vs adjusted_rand_s
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_db, x='epsi', y='Adjusted Rand Index Score', marker='o')
    plt.title('Adjusted_rand vs  epsi: db')
    plt.xlabel('Epsi')
    plt.ylabel('Adjusted_rand_s')
    plt.grid(True)
    plt.show()

    # Gráfico 6: Affinity Nº Clusters
    dampings = np.linspace(0.5, 0.9, 10)
    n_clusters_affinity = []

    for damping in dampings:
        (labels_aff, silhouette_aff,
         adj_rand_index_aff, n_clusters) = affinity(array_best_features, labels[:, 1],
                                                                              damping)
        n_clusters_affinity.append(n_clusters)

    plt.figure(figsize=(10, 6))
    plt.plot(dampings, n_clusters_affinity, marker='o')
    plt.title('N. de Clusters estimados pelo Affinity em Relação ao Damping')
    plt.xlabel('Damping')
    plt.ylabel('Número de Clusters')
    plt.grid(True)
    plt.show()

    # Gráfico 13 - recall_s_k-means
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_k_filtered, x='n_clusters', y='Recall Score',hue='i', marker='o')
    plt.title('Recall across diferent Clusters')
    plt.xlabel('N.º Clusters')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.show()

    # Gráfico 14 - recall_s_dbscan (no gráfico deles também só aparecem 5, por isso está ok)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_db, x='epsi', y='Recall Score', hue='i', marker='o')
    plt.title('Recall across diferent Epsis')
    plt.xlabel('Epsi')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.show()

    # Gráfico 15 - rand_ind_s_k-means
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_k_filtered, x='n_clusters', y='Rand Index Score', hue='i', marker='o')
    plt.title('Rand Index Score across diferent Clusters')
    plt.xlabel('N.º Clusters')
    plt.ylabel('Rand Index Score')
    plt.grid(True)
    plt.show()

    # Gráfico 16 - rand_ind_s_dbscan
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_db, x='epsi', y='Rand Index Score', hue='i', marker='o')
    plt.title('Rand Index Score across diferent Epsis')
    plt.xlabel('Epsi')
    plt.ylabel('Rand Index Score')
    plt.grid(True)
    plt.show()

    # Gráfico 17 - precision_s_k-means
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_k_filtered, x='n_clusters', y='Precision Score', hue='i', marker='o')
    plt.title('Precision across diferent Clusters')
    plt.xlabel('N.º Clusters')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()

    # Gráfico 18 - precision_s_dbscan
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_db, x='epsi', y='Precision Score', hue='i', marker='o')
    plt.title('Precision across diferent Epsis')
    plt.xlabel('Epsi')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()

    # Gráfico 19 - precision_s_k-means
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_k_filtered, x='n_clusters', y='F1 Score', hue='i', marker='o')
    plt.title('F1 Score across diferent Clusters')
    plt.xlabel('N.º Clusters')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.show()

    # Gráfico 20 - f1_s_k-means
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_db, x='epsi', y='F1 Score', hue='i', marker='o')
    plt.title('F1 Score across diferent Epsis')
    plt.xlabel('Epsi')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.show()

    # Gráfico 21 - adj_rand_ind_s_k-means
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_k_filtered, x='n_clusters', y='Adjusted Rand Index Score', hue='i', marker='o')
    plt.title('Adjusted Rand Index Score across diferent Clusters')
    plt.xlabel('N.º Clusters')
    plt.ylabel('Adjusted Rand Index Score')
    plt.grid(True)
    plt.show()

    # Gráfico 22 - adj_rand_ind_s_dbscan
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_db, x='epsi', y='Adjusted Rand Index Score', hue='i', marker='o')
    plt.title('Adjusted Rand Index Score across diferent Epsis')
    plt.xlabel('Epsi')
    plt.ylabel('Adjusted Rand Index Score')
    plt.grid(True)
    plt.show()


Main()



