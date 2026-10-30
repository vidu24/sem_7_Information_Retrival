# NewsLens AI — Fake News Detection & Intelligent News Retrieval

A Streamlit implementation of the Assignment 6 project. It combines supervised text classification with semantic retrieval and unsupervised corpus analysis.

## Features

- Article ingestion: paste text, import CSV, or fetch a public article URL.
- SQLite storage with URL duplicate prevention.
- Fake-news risk classifier: TF-IDF (unigrams + bigrams) and Logistic Regression.
- Confidence scores and explicit uncertainty messaging.
- Semantic search: Sentence Transformers embeddings with FAISS when installed; TF-IDF/cosine fallback otherwise.
- Clustering, topic keywords, PCA visualization, and Isolation Forest anomaly flags.
- CLIP text-to-image retrieval over user-uploaded article images.
- Retrieval-grounded news exploration.
- Evaluation metrics from a stratified holdout when the dataset is large enough.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Dataset format

Import a CSV from **Add / collect news → Import CSV**. Select:
- a text/content column (required),
- optionally a headline column,
- optionally a publisher/source column,
- a label column (required for supervised training).

Accepted labels:
- REAL: `real`, `true`, `reliable`, `1`, `genuine`
- FAKE: `fake`, `false`, `misleading`, `0`, `unreliable`

A common dataset may use columns named `title`, `text`, and `label`; map them in the import screen. Check the dataset's own label convention before mapping. Dataset licenses and terms must be followed.

## Run the project

1. Import a properly labeled CSV dataset.
2. Open **Train classifier** and train the model.
3. Paste an article in **Analyse an article** to screen it.
4. Add a larger corpus and use **Intelligent search**.
5. Use **Topics & anomalies** for KMeans clusters, LDA topic keywords, PCA visualization, and anomaly flags.
6. Use **News assistant** to retrieve relevant passages from the local collection.

## Important implementation notes

- The model is a baseline, not a production fact-checker. Dataset bias, publisher leakage, topic leakage, class imbalance, and distribution shift can invalidate apparent performance.
- The confidence score is the classifier's class probability, not the probability that a claim is true.
- `FAKE` means the text resembles the FAKE examples in the training dataset. It does not prove that the article is false.
- Anomaly detection identifies unusual patterns in the current corpus, not fake news.
- Semantic similarity is relevance evidence, not truth evidence.
- URL extraction may fail on paywalls, dynamic pages, bot protections, or sites that disallow automated access. Paste article text in those cases.
- CLIP image retrieval is available in the Image retrieval (CLIP) tab. It downloads the openai/clip-vit-base-patch32 model on first use; uploaded images are processed in memory and are not stored in SQLite. Do not treat an image match as verification.
- The conversational module is extractive retrieval: it displays retrieved article content rather than fabricating an answer.

## Suggested datasets

Use a dataset you are permitted to access, such as LIAR or FakeNewsNet (subject to its access conditions and license). Inspect the label definitions and split by source/time where possible. Do not combine train and test records before evaluation.

## Architecture

`Streamlit UI → Ingestion → SQLite → ML classifier / semantic index / clustering & anomaly analysis → results`

## Suggested next improvements

- Add claim extraction and evidence retrieval from trusted fact-checking sources.
- Add source reputation as a separately auditable feature rather than a hidden proxy.
- Add temporal/source-held-out evaluation.
- Add CLIP image embeddings and a multimodal index for article-associated images.
- Add a cross-encoder reranker and ranking metrics (nDCG@K, MRR, Recall@K).
- Add a proper RAG answer generator with citations to retrieved articles and abstention.
