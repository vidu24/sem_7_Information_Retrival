# Practical Assignment 5 — Intelligent Document Search Using Unsupervised Learning

## Objective
Build a document search system that discovers document groups and topics without manually labeled training data, flags possible anomalies, and ranks documents by semantic relevance.

## Dataset
The included `documents.csv` contains **25 short synthetic demonstration documents**: six each in AI, cybersecurity, databases, and networking, plus one unrelated gardening document. The `subject` column is reference metadata only and is never passed to K-Means, LDA, or the embedding model. `queries.json` contains five evaluation queries and manually curated relevant document IDs. These relevance sets are illustrative, not independently validated annotations.

## Methodology
1. Load the documents and create 384-dimensional normalized semantic vectors using the pretrained `all-MiniLM-L6-v2` sentence transformer.
2. Cluster vectors using K-Means with `k=4`, `random_state=42` and 20 initializations. Clusters are numeric IDs; their correspondence to subject areas is inferred by inspecting membership.
3. Use CountVectorizer unigram/bigram counts and Latent Dirichlet Allocation with four topics; output eight top terms per topic.
4. Apply PCA to reduce embeddings to two dimensions and draw a labeled scatter plot.
5. Compute Euclidean distances to assigned K-Means centroids and mark the farthest 5% of documents as anomaly **candidates**. A gardening document is included as a planted anomaly, but the detector is not guaranteed to flag it.
6. Encode each query with the same embedding model; rank all documents by cosine similarity. Show document ID, title, score, and cluster.
7. Compare the top five results against the supplied relevance judgments; compute Precision@5, Recall@5, and F1@5. Compute cosine silhouette score as a clustering diagnostic.

## Evaluation calculations
For each query, let TP be the number of relevant documents retrieved in the top five, and let R be the number of relevant documents in the judgment set.

- Precision@5 = TP / 5
- Recall@5 = TP / R
- F1@5 = 2 × Precision@5 × Recall@5 / (Precision@5 + Recall@5), defined as zero when the denominator is zero.
- Macro scores = arithmetic mean of per-query scores.

**Illustrative calculation (not an observed result):** If 3 of 5 returned documents are relevant and the relevance set has 4 documents, Precision@5 = 3/5 = 0.60; Recall@5 = 3/4 = 0.75; F1@5 = 0.667.

## Observations and results
The sample collection is designed so AI, cybersecurity, databases and networking documents share domain-specific vocabulary. The gardening document has unrelated content. Actual cluster assignments, topics, anomaly candidates, retrieval ranks, silhouette score, and per-query metrics must be read from `outputs/results.json` **after running the program**. No measured performance claims are made before execution.

## Limitations
- The sample collection is small and synthetic, so results may not generalize to full research papers.
- Four clusters and four LDA topics are preselected rather than automatically estimated.
- K-Means can place a genuine outlier into an ordinary cluster; a 95th-percentile distance threshold also flags a fixed fraction of the collection, even without true anomalies.
- PCA's two-dimensional layout distorts some high-dimensional distances.
- LDA topics and embedding clusters are different models and need not align.
- Retrieval judgments are manually specified and incomplete; metrics depend on these choices.
- The pretrained embedding model requires an initial internet download; CPU execution may take time.

## Conclusion
The implementation combines pretrained semantic embeddings with unsupervised clustering, LDA topic discovery, dimensionality reduction, anomaly screening and cosine-similarity search. Relevance judgments provide a reproducible but small-scale retrieval evaluation. Broader datasets, human judgments, model comparisons and calibrated outlier detection would improve validity.

## Submission outputs and screenshot checklist
After `python main.py`, capture the console output showing document count, LDA topics, anomaly candidates, search results and P@5/R@5/F1@5. Include `outputs/clusters.png` for the PCA plot, `outputs/document_assignments.csv` for cluster/topic assignments, `outputs/results.json` for calculations and `outputs/embeddings.npy` as the vector representation. Add these screenshots to the report before submitting if required.
