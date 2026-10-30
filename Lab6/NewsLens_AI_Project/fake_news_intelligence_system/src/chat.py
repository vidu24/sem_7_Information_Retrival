from .retrieval import SemanticIndex

def answer_question(question, articles, index=None):
    if not articles:
        return "There are no articles in the collection yet. Add articles first."
    texts = [f"{a.get('title','')}. {a.get('text','')}" for a in articles]
    index = index or SemanticIndex()
    index.build(texts)
    hits = index.search(question, k=3)
    if not hits:
        return "I could not retrieve a relevant article."
    output = ["Retrieved evidence (extractive; not an independent fact-check):"]
    for hit in hits:
        article = articles[hit["index"]]
        snippet = hit["text"][:650]
        output.append(f"\n• {article.get('title','Untitled')} (similarity: {hit['score']:.3f})\n  {snippet}")
    output.append("\nThe collection may be incomplete or biased. Verify important claims against independent primary sources.")
    return "\n".join(output)
