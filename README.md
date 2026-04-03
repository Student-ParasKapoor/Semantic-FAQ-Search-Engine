# Semantic FAQ Search Engine 🔍

## 📌 Overview

This project is a semantic FAQ search engine that retrieves the most relevant question from a dataset based on user input. It addresses the limitations of traditional keyword-based search by leveraging modern NLP techniques to understand the **meaning** of queries rather than just matching words.

---

## 🚀 Features

- 🔤 Keyword-based search using TF-IDF
- 🧠 Semantic search using word embeddings
- ⚡ Advanced contextual search using sentence transformers
- 📊 Comparison of multiple NLP approaches
- 🧩 Modular and extensible design

---

## 🧠 Approaches Implemented

### 1. TF-IDF + Cosine Similarity
- Converts text into sparse vectors based on word importance
- Works well for exact keyword matching
- ❌ Fails when synonyms or rephrasing are used

---

### 2. Word Embeddings (GloVe via Gensim)
- Represents words as dense vectors capturing semantic meaning
- Sentence vectors created via averaging word embeddings
- ✅ Better semantic understanding than TF-IDF  
- ❌ Loses word importance and context

---

### 3. Sentence Transformers (Final Model) 🚀
- Uses pretrained transformer models to generate sentence embeddings
- Captures context, word relationships, and intent
- ✅ State-of-the-art semantic search performance  
- ✅ Handles rephrasing, synonyms, and context effectively  

---

## 🛠️ Tech Stack

- Python
- scikit-learn
- Gensim
- sentence-transformers
- NumPy

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```text

📂 Semantic-FAQ-Search/
│
├── 📄 tfidf_search.py                     # Version 1: TF-IDF
├── 📄 embedding_search.py                 # Version 2: Word Embeddings
├── 📄 sentence_transformer_search.py      # Version 3: Final Model
├── 📄 requirements.txt
└── 📄 README.md  

```
---

## ▶️ Usage

Run any version:

```bash
python tfidf_search.py
python embedding_search.py
python sentence_transformer_search.py

```
---

# 🧪 Example

**Input Query:**
How can I change my password?

**Output (Sentence Transformer):**
Best matching FAQ: How can I reset my password?

## 📊 Key Learnings

- TF-IDF is fast but lacks semantic understanding  
- Word embeddings improve meaning but lose context  
- Sentence transformers provide the best results by capturing full sentence semantics 

## ⚠️ Limitations

- Small dataset limits evaluation  
- Sentence transformers are slower than TF-IDF  

## 🔮 Future Improvements

- Use larger FAQ datasets  
- Implement top-k results instead of single match  
- Integrate with RAG (Retrieval-Augmented Generation) systems  

## 👨‍💻 Author

Built as part of a hands-on NLP learning journey focusing on deep understanding of semantic search systems.