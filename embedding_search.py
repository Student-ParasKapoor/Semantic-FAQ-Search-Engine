import gensim.downloader as api
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

model = api.load("glove-wiki-gigaword-50")

def sentence_to_vector(sentence):
    words = re.findall(r'\b\w+\b', sentence.lower())
    
    word_vectors = [model[word] for word in words if word in model]

    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    
    return np.mean(word_vectors, axis=0)

faqs = [
    "How to reset password?",
    "How to change email?",
    "What is refund policy?"
]

faq_vectors = np.array([sentence_to_vector(faq) for faq in faqs])
query_vector = sentence_to_vector("how can i change my password?")
similarities = cosine_similarity([query_vector], faq_vectors)[0]
best_index = np.argmax(similarities)
best_match = faqs[best_index]
print(f"Best matching FAQ: {best_match}")