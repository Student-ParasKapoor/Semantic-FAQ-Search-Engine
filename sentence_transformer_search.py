from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")
faqs = [
    "How to reset password?",
    "How to change email?",
    "What is refund policy?"
]

faq_vectors = model.encode(faqs)
query = "how can i change my password?"
query_vector = model.encode(query)
similarities = cosine_similarity([query_vector], faq_vectors)[0]
best_index = similarities.argmax()
best_match = faqs[best_index]
print(f"Best matching FAQ: {best_match}")