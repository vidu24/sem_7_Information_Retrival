import re
import requests
from bs4 import BeautifulSoup

def clean_text(text):
    text = re.sub(r"\s+", " ", str(text or ""))
    return text.strip()

def extract_article(url, timeout=15):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsResearchBot/1.0)"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    title_tag = soup.find("meta", property="og:title")
    title = (title_tag.get("content") if title_tag else None) or (soup.title.get_text(" ", strip=True) if soup.title else url)
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    body = clean_text(" ".join(p for p in paragraphs if len(p) > 35))
    if len(body) < 100:
        body = clean_text(soup.get_text(" ", strip=True))
    if len(body) < 80:
        raise ValueError("Could not extract enough article text. Paste the article text instead.")
    return {"title": clean_text(title), "text": body, "source": url.split("/")[2], "url": url}
