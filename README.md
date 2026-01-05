# Unsupervised Image Clustering (PCA / t-SNE / Isomap)

This project explores **unsupervised clustering of grayscale images** using multiple feature extraction and clustering techniques.

It combines:
- **Dimensionality Reduction:** PCA, t-SNE, Isomap
- **Feature Selection:** ANOVA (Select best features)
- **Clustering Algorithms:** K-Means, DBSCAN, Affinity Propagation
- **Evaluation Metrics:** Silhouette Score, Adjusted Rand Index (ARI), and external clustering indices
- **HTML Reports:** Cluster visualizations showing images grouped per cluster

## Context
Academic project developed for the course **Aprendizagem Automatizada I (Machine Learning I)**.

## Dataset
- `images/` contains indexed grayscale images (`0.png` to `562.png`)
- `labels.txt` provides known labels for part of the dataset (used for evaluation and feature selection)

## Project Structure
- `TP2.py` — main pipeline (feature extraction, clustering, evaluation, plots)
- `tp2_aux.py` — helper functions to load images and generate HTML cluster reports
- `labels.txt` — (image_id, label)
- `images/` — image dataset

## How it works (high level)
1. Load images as vectors (flattened pixels)
2. Compute multiple reduced representations (PCA, t-SNE, Isomap)
3. Concatenate features and select best ones (ANOVA)
4. Run clustering (K-Means / DBSCAN / Affinity Propagation)
5. Evaluate using Silhouette + ARI (+ external metrics)
6. Generate HTML reports with clustered images

## How to run
```bash
pip install -r requirements.txt
python TP2.py
