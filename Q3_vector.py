import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east-1"
INDEX_NAME = "llama-text-embed-v2-index"
pc = Pinecone(api_key=PINECONE_API_KEY)



index = pc.Index(INDEX_NAME)

df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "text": [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is amazing!",
        "Pinecone is a great vector database.",
        "Embeddings help in NLP tasks.",
        "FastAPI is great for backend development."
    ]
})

model = SentenceTransformer("BAAI/bge-large-en")

def generate_embeddings(texts):
    return model.encode(texts, convert_to_numpy=True).tolist()

for i in range(0, len(df), 100):
    batch = df.iloc[i:i + 100]
    vectors = generate_embeddings(batch["text"].tolist())

    pinecone_data = [(str(batch.iloc[j]["id"]), vectors[j]) for j in range(len(batch))]
    index.upsert(vectors=pinecone_data)

print("Data uploaded to Pinecone")

def fetch_sample_embeddings():
    sample_ids = [str(i) for i in df["id"].tolist()]
    results = index.fetch(ids=sample_ids)

    print("Sample Embeddings:")
    for _id, data in results.vectors.items():
        print(f"ID: {_id} : {data.values[:5]}")



fetch_sample_embeddings()


