from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
faqs = [
    "How to reset password?",
    "How to change email?",
    "What is refund policy?"
]
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faqs)
query = "how can i change my password?"
query_vector = vectorizer.transform([query]) 
similarities = cosine_similarity(query_vector, faq_vectors)
best_index = similarities[0].argmax() 
best_match = faqs[best_index]
print(f"Best matching FAQ: {best_match}")