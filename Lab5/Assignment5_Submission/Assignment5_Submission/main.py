"""Assignment 5: Unsupervised intelligent document search.
Run: pip install -r requirements.txt && python main.py
SentenceTransformer downloads its model on first run.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

# Load demonstration documents. Labels are NOT used for unsupervised training.
with (ROOT / 'documents.csv').open(encoding='utf-8', newline='') as f:
    docs = list(csv.DictReader(f))
texts = [d['text'] for d in docs]
ids = [d['id'] for d in docs]

# Embed documents using a pretrained semantic sentence embedding model.
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
X = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
np.save(OUT / 'embeddings.npy', X)

# K-means groups the four main technical subjects; anomaly is a candidate outlier.
k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=20)
clusters = km.fit_predict(X)
centroids = km.cluster_centers_
distances = np.linalg.norm(X - centroids[clusters], axis=1)
# Flag the farthest 5% as candidates; this is a heuristic, not ground truth.
cutoff = float(np.quantile(distances, 0.95))
anomalies = [i for i, d in enumerate(distances) if d >= cutoff]

# LDA discovers topics from word-count features, independently of supplied labels.
cv = CountVectorizer(stop_words='english', max_df=0.95, min_df=1, ngram_range=(1, 2))
counts = cv.fit_transform(texts)
lda = LatentDirichletAllocation(n_components=k, random_state=42, max_iter=60)
topic_weights = lda.fit_transform(counts)
terms = cv.get_feature_names_out()
topics = {f'Topic {t}': [terms[j] for j in np.argsort(row)[-8:][::-1]]
          for t, row in enumerate(lda.components_)}

# Visualize embeddings in two principal components.
coords = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(9, 6))
for c in range(k):
    ix = np.where(clusters == c)[0]
    plt.scatter(coords[ix, 0], coords[ix, 1], label=f'Cluster {c}', s=55)
for i in anomalies:
    plt.scatter(coords[i, 0], coords[i, 1], s=190, facecolors='none',
                edgecolors='black', linewidths=2)
for i, doc_id in enumerate(ids):
    plt.annotate(doc_id, coords[i], xytext=(3, 3), textcoords='offset points', fontsize=7)
plt.title('PCA projection of document embeddings (circled: anomaly candidates)')
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(); plt.tight_layout()
plt.savefig(OUT / 'clusters.png', dpi=180)
plt.close()

# Retrieve documents by semantic cosine similarity; clusters aid interpretation.
def search(query, top_k=5):
    q = model.encode([query], normalize_embeddings=True)
    similarities = cosine_similarity(q, X)[0]
    order = np.argsort(-similarities)[:top_k]
    return [{'id': ids[i], 'score': round(float(similarities[i]), 4),
             'cluster': int(clusters[i]), 'title': docs[i]['title'],
             'explanation': 'Semantic embedding cosine similarity to query'} for i in order]

# Manually curated relevance judgments are only for evaluation, not model training.
with (ROOT / 'queries.json').open(encoding='utf-8') as f:
    judgments = json.load(f)
metrics = []
for entry in judgments:
    results = search(entry['query'], top_k=5)
    relevant = set(entry['relevant_ids'])
    hits = sum(r['id'] in relevant for r in results)
    precision = hits / len(results)
    recall = hits / len(relevant) if relevant else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    metrics.append({'query': entry['query'], 'precision_at_5': precision,
                    'recall_at_5': recall, 'f1_at_5': f1, 'results': results})

report = {'documents': len(docs), 'clusters': k, 'embedding_dimensions': int(X.shape[1]),
          'silhouette_cosine': float(silhouette_score(X, clusters, metric='cosine')),
          'topics': topics, 'anomaly_threshold': cutoff,
          'anomalies': [{'id': ids[i], 'title': docs[i]['title'], 'distance': float(distances[i])}
                        for i in anomalies],
          'cluster_members': {str(c): [ids[i] for i in range(len(ids)) if clusters[i] == c]
                              for c in range(k)},
          'evaluations': metrics,
          'macro_precision_at_5': float(np.mean([m['precision_at_5'] for m in metrics])),
          'macro_recall_at_5': float(np.mean([m['recall_at_5'] for m in metrics])),
          'macro_f1_at_5': float(np.mean([m['f1_at_5'] for m in metrics]))}
(OUT / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
with (OUT / 'document_assignments.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'title', 'cluster', 'dominant_lda_topic', 'centroid_distance', 'anomaly_candidate'])
    for i, d in enumerate(docs):
        writer.writerow([ids[i], d['title'], int(clusters[i]), int(np.argmax(topic_weights[i])),
                         round(float(distances[i]), 5), i in anomalies])

print('Documents:', len(docs), '| Embedding shape:', X.shape)
print('Silhouette (cosine):', round(report['silhouette_cosine'], 4))
print('LDA topics:', topics)
print('Anomaly candidates:', report['anomalies'])
for item in metrics:
    print('\nQuery:', item['query'])
    for r in item['results']:
        print(f"  {r['id']}: {r['title']} (similarity={r['score']}, cluster={r['cluster']})")
    print('P@5:', round(item['precision_at_5'], 3),
          'R@5:', round(item['recall_at_5'], 3), 'F1@5:', round(item['f1_at_5'], 3))
print('\nOutputs saved in', OUT)

# Optional interactive terminal search interface.
if __name__ == '__main__':
    while True:
        try:
            query = input('\nEnter search query (blank to exit): ').strip()
        except EOFError:
            break
        if not query:
            break
        for r in search(query):
            print(f"{r['id']} | {r['title']} | similarity={r['score']} | cluster={r['cluster']}")
