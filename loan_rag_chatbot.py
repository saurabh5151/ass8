# loan_rag_chatbot.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai  # Optional for GPT

# Load CSV
df = pd.read_csv("Training Dataset.csv")
text_data = df.astype(str).apply(lambda row: ' | '.join(row), axis=1).tolist()

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(text_data, show_progress_bar=True)

# FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Retrieval function
def retrieve_context(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [text_data[i] for i in indices[0]]

# OpenAI for answer generation (optional, requires API key)
def generate_answer(query, context_docs):
    context = "\n".join(context_docs)
    prompt = f"""You are a helpful assistant. Based on the following data:\n\n{context}\n\nAnswer this question: {query}"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response['choices'][0]['message']['content']
