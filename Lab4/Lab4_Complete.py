"""
Practical Assignment 4: Search Engine Ranking and Evaluation
Mini search engine for a college library.

Models:
1. TF-IDF score
2. Vector Space Model (TF-IDF cosine similarity)
3. Probabilistic Retrieval Model (BM25)
4. Binary Independence Model (BIM, RSJ weights with relevance feedback)

Evaluation:
Precision@5, Recall@5, F1@5, MRR@5, AP@5 (MAP across test queries),
and NDCG@5.

"""

import math
import re
from collections import Counter

# ---------------------------------------------------------------------
# 1. Dataset: at least 10 college-library documents
# ---------------------------------------------------------------------
documents = {
    "D1": "Machine learning methods for medical diagnosis and disease detection",
    "D2": "Healthcare data analysis using artificial intelligence",
    "D3": "History and development of computer networks",
    "D4": "Deep learning techniques for disease prediction",
    "D5": "Database management systems and SQL fundamentals",
    "D6": "Machine learning applications in modern healthcare",
    "D7": "Introduction to data structures and algorithms",
    "D8": "Natural language processing for electronic health records",
    "D9": "Fundamentals of quantum computing and quantum physics",
    "D10": "Predictive analytics in healthcare using deep learning",
    "D11": "Computer vision applications in medical image analysis",
    "D12": "Cybersecurity and intrusion detection in computer networks",
}

stopwords = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in",
    "into", "is", "of", "on", "or", "the", "to", "using", "with"
}

def tokenize(text):
    """Lowercase text, extract word tokens, and remove common stopwords."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stopwords]

doc_tokens = {doc_id: tokenize(text) for doc_id, text in documents.items()}
N = len(documents)
df = Counter()
for tokens in doc_tokens.values():
    df.update(set(tokens))
avgdl = sum(len(t) for t in doc_tokens.values()) / N

# ---------------------------------------------------------------------
# 2. Test queries and illustrative relevance judgments
# Relevant document IDs should be reviewed by human evaluators.
# ---------------------------------------------------------------------
test_queries = {
    "Q1": {
        "query": "machine learning applications in healthcare",
        "relevant": {"D1", "D2", "D4", "D6", "D8", "D10"},
    },
    "Q2": {
        "query": "computer networks and cybersecurity",
        "relevant": {"D3", "D12"},
    },
    "Q3": {
        "query": "quantum computing physics",
        "relevant": {"D9"},
    },
}

def idf(term):
    """Smoothed IDF prevents zero or undefined weights."""
    return math.log(1 + (N + 1) / (df[term] + 1))

def tfidf_vector(tokens):
    """Create a normalized TF-IDF vector represented as a dictionary."""
    tf = Counter(tokens)
    vec = {term: (1 + math.log(count)) * idf(term) for term, count in tf.items()}
    norm = math.sqrt(sum(weight * weight for weight in vec.values()))
    if norm:
        vec = {term: weight / norm for term, weight in vec.items()}
    return vec

doc_vectors = {doc_id: tfidf_vector(tokens) for doc_id, tokens in doc_tokens.items()}

def tfidf_score(query_tokens, doc_id):
    """Unnormalized TF-IDF matching score for query terms."""
    tf = Counter(doc_tokens[doc_id])
    return sum((1 + math.log(tf[t])) * idf(t) * idf(t)
               for t in set(query_tokens) if tf[t] > 0)

def vsm_score(query_tokens, doc_id):
    """Cosine similarity between normalized TF-IDF document and query vectors."""
    qvec = tfidf_vector(query_tokens)
    dvec = doc_vectors[doc_id]
    return sum(qvec.get(t, 0.0) * dvec.get(t, 0.0) for t in qvec)

def bim_score(query_tokens, doc_id, relevant_docs):
    """
    Binary Independence Model with Robertson-Sparck Jones term weights.
    Uses query-specific relevance feedback: relevant_docs are the judged
    relevant documents for this query. Scores use binary term presence.
    """
    tokens = set(doc_tokens[doc_id])
    R = len(relevant_docs)
    score = 0.0
    for term in set(query_tokens):
        n = df[term]
        r = sum(1 for d in relevant_docs if term in set(doc_tokens[d]))
        # Additive smoothing keeps probabilities finite.
        p = (r + 0.5) / (R + 1)
        q = (n - r + 0.5) / (N - R + 1)
        weight = math.log((p * (1 - q)) / (q * (1 - p)))
        if term in tokens:
            score += weight
    return score

models = {
    "TF-IDF": tfidf_score,
    "VSM cosine": vsm_score,
   
}

def rank_documents(query, model_name, relevant_docs):
    """Return all documents sorted by descending score, with stable ID tie-break."""
    qtokens = tokenize(query)
    if model_name == "BIM":
        scorer = lambda doc_id: bim_score(qtokens, doc_id, relevant_docs)
    else:
        scorer = lambda doc_id: models[model_name](qtokens, doc_id)
    scored = [(doc_id, scorer(doc_id)) for doc_id in documents]
    return sorted(scored, key=lambda item: (-item[1], item[0]))

def evaluate(ranked, relevant_docs, k=5):
    """Calculate binary-relevance metrics at cutoff k."""
    retrieved = ranked[:k]
    rel = [1 if doc_id in relevant_docs else 0 for doc_id, _ in retrieved]
    hits = sum(rel)
    precision = hits / k if k else 0.0
    recall = hits / len(relevant_docs) if relevant_docs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    reciprocal_rank = next((1 / (i + 1) for i, value in enumerate(rel) if value), 0.0)

    hits_so_far = 0
    precision_sum = 0.0
    for i, value in enumerate(rel, start=1):
        if value:
            hits_so_far += 1
            precision_sum += hits_so_far / i
    ap = precision_sum / len(relevant_docs) if relevant_docs else 0.0

    dcg = sum(value / math.log2(i + 2) for i, value in enumerate(rel))
    ideal_count = min(len(relevant_docs), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_count))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "P@5": precision, "Recall@5": recall, "F1@5": f1,
        "MRR@5": reciprocal_rank, "AP@5": ap, "NDCG@5": ndcg
    }

def main():
    model_names = ["TF-IDF", "VSM cosine", "BIM"]
    aggregate = {name: [] for name in model_names}

    print("=" * 88)
    print("COLLEGE LIBRARY MINI SEARCH ENGINE")
    print(f"Documents: {N} | Test queries: {len(test_queries)}")
    print("=" * 88)

    for qid, item in test_queries.items():
        query, relevant = item["query"], item["relevant"]
        print(f"\n{qid}: {query}")
        print("Judged relevant documents:", ", ".join(sorted(relevant)))
        for model_name in model_names:
            ranked = rank_documents(query, model_name, relevant)
            print(f"\n  {model_name} — Top 5")
            for rank, (doc_id, score) in enumerate(ranked[:5], start=1):
                label = "RELEVANT" if doc_id in relevant else "NOT RELEVANT"
                print(f"    {rank}. {doc_id} | score={score:.4f} | {label} | {documents[doc_id]}")
            metrics = evaluate(ranked, relevant)
            aggregate[model_name].append(metrics)

    print("\n" + "=" * 88)
    print("AGGREGATE EVALUATION (mean over test queries)")
    print("=" * 88)
    metric_names = ["P@5", "Recall@5", "F1@5", "MRR@5", "AP@5", "NDCG@5"]
    print(f"{'Model':<14}" + "".join(f"{m:>11}" for m in metric_names))
    for model_name in model_names:
        averages = {
            m: sum(row[m] for row in aggregate[model_name]) / len(aggregate[model_name])
            for m in metric_names
        }
        print(f"{model_name:<14}" + "".join(f"{averages[m]:>11.3f}" for m in metric_names))

    print("The Probabilistic Ranking Principle ranks documents by decreasing estimated")
    print("probability of relevance to the query. BIM models binary")
    print("term presence and can use relevance judgments as feedback.")
    print("BIM scores are ranking scores, not calibrated probabilities.")


if __name__ == "__main__":
    main()
