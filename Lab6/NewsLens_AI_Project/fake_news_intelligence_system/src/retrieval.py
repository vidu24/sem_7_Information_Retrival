import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticIndex:
    """Sentence-Transformer embeddings with a TF-IDF fallback; FAISS is used when installed."""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = None
        self.faiss = None
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model_name)
        except Exception:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
            self.fallback = True
        try:
            import faiss
            self.faiss = faiss
        except Exception:
            pass
        self.texts, self.matrix, self.index = [], None, None

    def _encode(self, texts):
        if self.encoder:
            return np.asarray(self.encoder.encode(texts, normalize_embeddings=True), dtype="float32")
        return self.vectorizer.fit_transform(texts).toarray().astype("float32")

    def build(self, texts):
        self.texts = list(texts)
        if not self.texts:
            self.matrix = None
            return
        if self.encoder:
            self.matrix = self._encode(self.texts)
        else:
            self.matrix = self.vectorizer.fit_transform(self.texts).toarray().astype("float32")
        if self.faiss is not None and self.encoder:
            self.index = self.faiss.IndexFlatIP(self.matrix.shape[1])
            self.index.add(self.matrix)

    def search(self, query, k=5):
        if self.matrix is None or not self.texts:
            return []
        if self.encoder:
            q = np.asarray(self.encoder.encode([query], normalize_embeddings=True), dtype="float32")
        else:
            q = self.vectorizer.transform([query]).toarray().astype("float32")
        if self.index is not None:
            scores, ids = self.index.search(q, min(k, len(self.texts)))
            pairs = zip(ids[0], scores[0])
        else:
            sims = cosine_similarity(q, self.matrix)[0]
            pairs = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:k]
        return [{"index": int(i), "score": float(s), "text": self.texts[int(i)]}
                for i,s in pairs if int(i) >= 0]
