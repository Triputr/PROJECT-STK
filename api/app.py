from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import itertools
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)
CORS(app)

# ==================================================
# LOAD DATA PICKLE
# ==================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

with open(os.path.join(DATA_DIR, "paper.pkl"), "rb") as f:
    paper = pickle.load(f)

with open(os.path.join(DATA_DIR, "processed_paper.pkl"), "rb") as f:
    processed_paper = pickle.load(f)

with open(os.path.join(DATA_DIR, "words.pkl"), "rb") as f:
    words = pickle.load(f)

with open(os.path.join(DATA_DIR, "thesaurus.pkl"), "rb") as f:
    thesaurus = pickle.load(f)

print("Data berhasil dimuat")
print(f"Jumlah dokumen: {len(paper)}")

# ==================================================
# NLP SETUP
# ==================================================

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# ==================================================
# PREPROCESS QUERY
# ==================================================

def preprocess_query(query):
    query = query.lower()
    query = re.sub(r'[^a-zA-Z\s]', '', query)
    query = stopword.remove(query)

    tokens = query.split()
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens

# ==================================================
# SEARCH TANPA QUERY EXPANSION
# ==================================================

def search_without_expansion(init_query, top_n=10):

    query = preprocess_query(init_query)

    if not query:
        return []

    vectorizer = TfidfVectorizer(use_idf=True)

    x = [' '.join(query)]

    paper_tfidf = vectorizer.fit_transform(
        x + processed_paper
    )

    q = paper_tfidf[0]

    result = cosine_similarity(
        paper_tfidf,
        q
    )

    final = [
        [num, y[0], ' '.join(query)]
        for num, y in enumerate(result)
        if y[0] > 0.0
    ]

    final = sorted(
        final,
        key=lambda x: x[1],
        reverse=True
    )

    seen = set()
    new_result = []

    for item in final:
        if item[0] not in seen:
            seen.add(item[0])
            new_result.append(item)

    hasil = []

    for x in new_result[1:top_n+1]:

        doc = paper[x[0]-1]

        hasil.append({
            "judul": doc["Judul"],
            "url": doc["URL"],
            "score": round(float(x[1]), 4),
            "query": x[2]
        })

    return hasil

# ==================================================
# SEARCH DENGAN QUERY EXPANSION
# ==================================================

def search_with_expansion(init_query, top_n=10):

    query = preprocess_query(init_query)

    if not query:
        return []

    list_synonym = []

    for w in query:

        if w in thesaurus:
            list_synonym.append(thesaurus[w])

        else:
            list_synonym.append([w])

    qs = []

    for combo in itertools.product(*list_synonym):

        combo = [stemmer.stem(w) for w in combo]

        qs.append([' '.join(combo)])

    vectorizer = TfidfVectorizer(use_idf=True)

    max_result = []

    for x in qs:

        paper_tfidf = vectorizer.fit_transform(
            x + processed_paper
        )

        q = paper_tfidf[0]

        result = cosine_similarity(
            paper_tfidf,
            q
        )

        final = [
            [num, y[0], x[0]]
            for num, y in enumerate(result)
            if y[0] > 0.0
        ]

        max_result += final

    max_result = sorted(
        max_result,
        key=lambda x: x[1],
        reverse=True
    )

    seen = set()
    new_result = []

    for item in max_result:

        if item[0] not in seen:

            seen.add(item[0])
            new_result.append(item)

    hasil = []

    for x in new_result[1:top_n+1]:

        doc = paper[x[0]-1]

        hasil.append({
            "judul": doc["Judul"],
            "url": doc["URL"],
            "score": round(float(x[1]), 4),
            "query": x[2]
        })

    return hasil

# ==================================================
# API ENDPOINT
# ==================================================

@app.route("/")
def home():
    return jsonify({
        "message": "IR System API Running"
    })

@app.route("/api/search", methods=["GET"])
def search():

    query = request.args.get("q", "")
    mode = request.args.get("mode", "0")

    if query.strip() == "":
        return jsonify([])

    if mode == "1":
        hasil = search_with_expansion(query)
    else:
        hasil = search_without_expansion(query)

    return jsonify(hasil)

# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )