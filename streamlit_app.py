import streamlit as st
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from sentence_transformers import SentenceTransformer

FAQS = [
    "How can I reset my password?",
    "How do I change my account email address?",
    "What is the refund policy for purchases?",
    "How can I update my profile information?",
    "What should I do if I forgot my password?"
]

@st.cache_resource
def load_models():
    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectors = tfidf_vectorizer.fit_transform(FAQS)

    # Word Embeddings
    embedding_model = api.load("glove-wiki-gigaword-50")

    # Sentence Transformer
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    st_vectors = st_model.encode(FAQS)

    return {
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_vectors": tfidf_vectors,
        "embedding_model": embedding_model,
        "st_model": st_model,
        "st_vectors": st_vectors
    }


models = load_models()

def sentence_to_vector(sentence, model):
    words = re.findall(r'\b\w+\b', sentence.lower())
    vectors = [model[word] for word in words if word in model]

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


def run_tfidf(query):
    vectorizer = models["tfidf_vectorizer"]
    faq_vectors = models["tfidf_vectors"]

    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, faq_vectors)[0]

    return similarities


def run_embeddings(query):
    embedding_model = models["embedding_model"]

    query_vec = sentence_to_vector(query, embedding_model)
    faq_vecs = np.array([sentence_to_vector(faq, embedding_model) for faq in FAQS])

    similarities = cosine_similarity([query_vec], faq_vecs)[0]

    return similarities


def run_transformer(query):
    st_model = models["st_model"]
    st_vectors = models["st_vectors"]

    query_vec = st_model.encode(query)
    similarities = cosine_similarity(query_vec.reshape(1, -1), st_vectors)[0]

    return similarities


def main():
    st.set_page_config(page_title="Semantic FAQ Search", layout="centered")

    st.title("Semantic FAQ Search Engine 🔍")
    st.markdown("Compare different NLP approaches for semantic search.")

    model_choice = st.selectbox(
        "Choose Search Method:",
        ["TF-IDF", "Word Embeddings", "Sentence Transformers"]
    )

    query = st.text_input("Enter your question:")

    if query:
        if model_choice == "TF-IDF":
            similarities = run_tfidf(query)

        elif model_choice == "Word Embeddings":
            similarities = run_embeddings(query)

        else:
            similarities = run_transformer(query)

        # Top result
        best_index = similarities.argmax()
        best_match = FAQS[best_index]

        st.subheader("Best Matching FAQ:")
        st.success(best_match)

        # Show similarity score
        st.caption(f"Similarity Score: {similarities[best_index]:.4f}")

        # 🔥 Bonus: Show Top 3 Results
        st.subheader("Top 3 Matches:")
        top_indices = similarities.argsort()[-3:][::-1]

        for i in top_indices:
            st.write(f"- {FAQS[i]} (Score: {similarities[i]:.4f})")

if __name__ == "__main__":
    main()