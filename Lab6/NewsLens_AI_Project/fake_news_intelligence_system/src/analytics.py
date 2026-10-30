import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer

def analyse_corpus(texts, n_clusters=5):
    if len(texts) < 2:
        return {"message": "Add at least two articles to run corpus analysis."}
    vec = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    k = max(2, min(n_clusters, len(texts)))
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    terms = np.array(vec.get_feature_names_out())
    topics = {}
    for c in sorted(set(labels)):
        mean = np.asarray(X[labels == c].mean(axis=0)).ravel()
        topics[int(c)] = terms[np.argsort(mean)[-8:][::-1]].tolist()
    anomaly = IsolationForest(contamination="auto", random_state=42).fit_predict(X)
    lda = LatentDirichletAllocation(n_components=min(5, max(2, len(texts)//3)), random_state=42)
    doc_topics = lda.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())
    return {
        "clusters": labels.tolist(), "topics": topics,
        "anomalies": [int(x == -1) for x in anomaly],
        "pca": coords.tolist(),
        "dominant_topics": doc_topics.argmax(axis=1).tolist()
    }
