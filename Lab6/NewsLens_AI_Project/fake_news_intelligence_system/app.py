import io, re
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
from src.db import init_db, add_article, get_articles, count_articles, clear_articles
from src.ingest import extract_article
from src.ml import train_model, load_model, predict, normalize_label
from src.retrieval import SemanticIndex
from src.analytics import analyse_corpus
from src.chat import answer_question

st.set_page_config(page_title="NewsLens AI", page_icon="📰", layout="wide")
init_db()
st.title("📰 NewsLens AI")
st.caption("Fake-news risk screening · semantic news retrieval · topic discovery")
st.warning("Model outputs are research signals, not factual verdicts. Always verify claims with independent, reputable sources.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose module", ["Overview", "Add / collect news", "Train classifier", "Analyse an article", "Intelligent search", "Topics & anomalies", "Image retrieval (CLIP)", "News assistant", "Evaluation"])
    st.divider()
    st.metric("Articles in database", count_articles())
    st.caption("First run: add a labeled CSV dataset and train the classifier.")

articles = get_articles()
df_articles = pd.DataFrame(articles)

if page == "Overview":
    c1,c2,c3 = st.columns(3)
    c1.metric("Stored articles", len(articles))
    c2.metric("Classifier", "Trained" if load_model() else "Not trained")
    c3.metric("Retrieval", "Sentence Transformers / TF-IDF fallback")
    st.subheader("Project workflow")
    st.markdown("1. Import a labeled dataset or add article text/URLs.\\n2. Train a baseline TF-IDF + Logistic Regression classifier.\\n3. Screen an article and inspect its confidence.\\n4. Search the collection semantically and inspect related coverage.\\n5. Explore clusters, topics, and anomalous text patterns.")
    st.info("The bundled project does not pretend that a model trained on a tiny demo set is production-grade. Use a properly labeled dataset for meaningful evaluation.")

elif page == "Add / collect news":
    st.subheader("Collect articles")
    tab1,tab2,tab3 = st.tabs(["Paste article", "Import CSV", "Fetch URL"])
    with tab1:
        with st.form("paste"):
            title = st.text_input("Headline")
            body = st.text_area("Article text", height=220)
            source = st.text_input("Publisher / source")
            url = st.text_input("Article URL (optional)")
            label = st.selectbox("Known label (optional)", ["Unlabeled", "REAL", "FAKE"])
            submitted = st.form_submit_button("Save article")
        if submitted:
            if len(body.strip()) < 40: st.error("Please provide at least 40 characters of article text.")
            else:
                aid,new = add_article(title,body,source,url,"" if label=="Unlabeled" else label)
                st.success(f"Saved article #{aid}." if new else f"URL already exists (article #{aid}).")
    with tab2:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            data = pd.read_csv(up)
            st.write("Detected columns:", list(data.columns))
            title_col = st.selectbox("Headline column", ["(none)"] + list(data.columns))
            text_col = st.selectbox("Text/content column", list(data.columns))
            source_col = st.selectbox("Source column", ["(none)"] + list(data.columns))
            label_col = st.selectbox("Label column", ["(none)"] + list(data.columns))
            if st.button("Import rows"):
                n=0
                for _,r in data.iterrows():
                    body=str(r.get(text_col,""))
                    if len(body.strip()) < 20: continue
                    title=str(r.get(title_col,"")) if title_col!="(none)" else body[:100]
                    source=str(r.get(source_col,"")) if source_col!="(none)" else ""
                    lab=normalize_label(r.get(label_col,"")) if label_col!="(none)" else None
                    add_article(title,body,source,"",lab or "")
                    n+=1
                st.success(f"Imported {n} rows. Duplicate URLs are skipped when URLs are provided.")
    with tab3:
        url_in = st.text_input("Public article URL")
        if st.button("Fetch article"):
            try:
                item=extract_article(url_in)
                aid,new=add_article(item["title"],item["text"],item["source"],item["url"])
                st.success(f"Fetched and saved article #{aid}." if new else "This URL is already in the database.")
                st.write(item["title"])
                st.write(item["text"][:700]+"…")
            except Exception as e: st.error(f"Could not fetch article: {e}")
    st.divider()
    if articles:
        st.subheader("Stored articles")
        st.dataframe(pd.DataFrame(articles)[["id","title","source","label","created_at"]], use_container_width=True)
        if st.button("Delete all stored articles"):
            clear_articles(); st.rerun()

elif page == "Train classifier":
    st.subheader("Train fake-news risk classifier")
    labeled = [a for a in articles if normalize_label(a.get("label",""))]
    st.write(f"Labeled articles available: **{len(labeled)}**")
    st.caption("Accepted labels: REAL / TRUE / RELIABLE / 1 and FAKE / FALSE / MISLEADING / 0.")
    if st.button("Train TF-IDF + Logistic Regression"):
        try:
            data=pd.DataFrame(labeled)
            model,metrics=train_model(data,"text","label")
            st.session_state["metrics"]=metrics
            st.success("Model trained and saved.")
            st.json(metrics)
        except Exception as e: st.error(str(e))
    if load_model(): st.success("A trained model is available.")
    else: st.info("No trained model found. Import labeled examples before training.")

elif page == "Analyse an article":
    st.subheader("Article screening")
    title=st.text_input("Headline")
    text=st.text_area("Paste full article text",height=260)
    if st.button("Run screening"):
        model=load_model()
        if not model: st.error("Train the classifier first.")
        elif len(text.strip())<40: st.error("Enter a longer article.")
        else:
            result=predict(model, f"{title}. {text}")
            col1,col2=st.columns(2)
            col1.metric("Model signal", "Potentially misleading" if result["label"]=="FAKE" else "More consistent with REAL training examples")
            col2.metric("Model confidence", f"{result['confidence']:.1%}")
            st.progress(result["confidence"])
            st.json(result["probabilities"])
            st.warning(result["caution"])
            st.caption("Confidence is confidence in the model's learned class assignment, not the probability that the article is factually true.")

elif page == "Intelligent search":
    st.subheader("Semantic news search")
    query=st.text_input("Search by meaning, event, entity, or question")
    k=st.slider("Number of results",1,15,5)
    if st.button("Search collection") and query:
        if not articles: st.warning("Add articles first.")
        else:
            corpus=[f"{a['title']}. {a['text']}" for a in articles]
            with st.spinner("Building retrieval index…"):
                index=SemanticIndex(); index.build(corpus); hits=index.search(query,k)
            for h in hits:
                a=articles[h["index"]]
                st.markdown(f"### {a['title']}")
                st.caption(f"Source: {a.get('source') or 'Unknown'} · Similarity: {h['score']:.3f} · Label: {a.get('label') or 'Unlabeled'}")
                st.write(a["text"][:500]+("…" if len(a["text"])>500 else ""))
                if a.get("url"): st.markdown(f"[Open article]({a['url']})")
                st.divider()

elif page == "Topics & anomalies":
    st.subheader("Corpus exploration")
    if len(articles)<2: st.info("Add at least two articles.")
    else:
        n=st.slider("Maximum clusters",2,min(10,len(articles)),min(5,len(articles)))
        if st.button("Run corpus analysis"):
            texts=[f"{a['title']}. {a['text']}" for a in articles]
            result=analyse_corpus(texts,n)
            st.session_state["analysis"]=result
        result=st.session_state.get("analysis")
        if result and "topics" in result:
            rows=[]
            for i,a in enumerate(articles):
                rows.append({"Article":a["title"][:65],"Cluster":int(result["clusters"][i]),"Anomaly": "Flagged" if result["anomalies"][i] else "No"})
            st.dataframe(pd.DataFrame(rows),use_container_width=True)
            st.subheader("Cluster keywords")
            for c,words in result["topics"].items(): st.write(f"**Cluster {c}:** {', '.join(words)}")
            coords=result["pca"]
            plot=pd.DataFrame({"PC1":[p[0] for p in coords],"PC2":[p[1] for p in coords],"Cluster":[str(x) for x in result["clusters"]],"Article":[a["title"] for a in articles]})
            st.plotly_chart(px.scatter(plot,x="PC1",y="PC2",color="Cluster",hover_name="Article",title="PCA projection of article clusters"),use_container_width=True)
            st.caption("Anomaly detection flags unusual text patterns relative to this corpus; it does not establish misinformation.")


elif page == "Image retrieval (CLIP)":
    st.subheader("Text-to-image news retrieval")
    st.caption("Upload article-associated images and search them using a natural-language description.")
    query = st.text_input("Describe the image or scene")
    uploads = st.file_uploader("Upload images", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)
    if st.button("Search images"):
        if not query or not uploads:
            st.warning("Enter a query and upload at least one image.")
        else:
            try:
                from src.multimodal import ClipImageSearch
                images = [Image.open(f).convert("RGB") for f in uploads]
                with st.spinner("Loading CLIP and comparing images…"):
                    engine = ClipImageSearch()
                    hits = engine.search(query, images)
                cols = st.columns(min(3, len(hits)))
                for rank, hit in enumerate(hits):
                    with cols[rank % len(cols)]:
                        st.image(images[hit["index"]], use_container_width=True)
                        st.caption(f"Rank {rank+1} · CLIP similarity score: {hit['score']:.3f}")
                st.warning("CLIP measures text-image alignment. It does not verify the image's authenticity, date, location, or relationship to the article.")
            except Exception as e:
                st.error(f"CLIP could not run: {e}. Check that transformers and torch are installed and that the model can be downloaded.")

elif page == "News assistant":
    st.subheader("Ask questions about your collected news")
    q=st.text_input("Question")
    if st.button("Ask") and q:
        if not articles: st.warning("Add articles first.")
        else:
            with st.spinner("Retrieving supporting articles…"):
                st.text(answer_question(q,articles))
    st.caption("The assistant retrieves relevant article passages; it does not generate unsupported factual answers.")

elif page == "Evaluation":
    st.subheader("Classifier evaluation")
    metrics=st.session_state.get("metrics")
    if metrics:
        st.json(metrics)
        if "accuracy" in metrics:
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Accuracy",f"{metrics['accuracy']:.3f}")
            c2.metric("Fake precision",f"{metrics['precision_fake']:.3f}")
            c3.metric("Fake recall",f"{metrics['recall_fake']:.3f}")
            c4.metric("Fake F1",f"{metrics['f1_fake']:.3f}")
            cm=metrics.get("confusion_matrix")
            if cm:
                st.write("Confusion matrix (rows = actual REAL, FAKE; columns = predicted REAL, FAKE)")
                st.dataframe(pd.DataFrame(cm,index=["Actual REAL","Actual FAKE"],columns=["Predicted REAL","Predicted FAKE"]))
    else: st.info("Train a model in the Train classifier tab to calculate evaluation metrics.")
    st.markdown("**Recommended additional evaluation:** macro-F1, per-class precision/recall, ROC-AUC, PR-AUC, calibration, temporal holdout, and retrieval Precision@K / Recall@K / nDCG@K. Do not report metrics from the training set as test performance.")
