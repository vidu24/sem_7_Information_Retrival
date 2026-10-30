import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# Download required NLTK data resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Input text
text = ("The students are studying different subjects and they are enjoying "
        "their studies. The teachers are teaching the subjects very effectively.")

# 1. Tokenization (alphanumeric tokens only to remove punctuation)
tokens = [w for w in word_tokenize(text) if w.isalnum()]

# 2. Stop-word Removal (preserving case-lowered comparison)
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w.lower() not in stop_words]

# 3. Stemming (using Porter Stemmer)
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(w) for w in filtered_tokens]

# Helper function to map NLTK POS tags to WordNet POS tags for accurate lemmatization
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

# 4. Lemmatization (using WordNetLemmatizer with POS tags)
lemmatizer = WordNetLemmatizer()
pos_tags = nltk.pos_tag(filtered_tokens)
lemmatized_words = [
    lemmatizer.lemmatize(w.lower(), get_wordnet_pos(tag)) 
    for w, tag in pos_tags
]

# Display Results
print("=" * 70)
print("1. ORIGINAL TEXT:")
print(text)
print("=" * 70)

print("\n2. TOKENS:")
print(tokens)

print("\n3. AFTER STOP-WORD REMOVAL:")
print(filtered_tokens)

print("\n4. STEMMED WORDS (Porter Stemmer):")
print(stemmed_words)

print("\n5. LEMMATIZED WORDS (WordNet Lemmatizer with POS):")
print(lemmatized_words)

print("=" * 70)
print("6. COMPARISON: STEMMING vs. LEMMATIZATION")
print("=" * 70)
print(f"{'Token':<15} | {'Stemmed Form':<15} | {'Lemmatized Form':<15}")
print("-" * 50)
for original, stem, lemma in zip(filtered_tokens, stemmed_words, lemmatized_words):
    print(f"{original:<15} | {stem:<15} | {lemma:<15}")