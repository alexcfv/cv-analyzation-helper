import chromadb
import uuid

class VectorStore:
    def __init__(self, client):
        self.collection = client.get_or_create_collection("resumes")

    def add_documents(self, documents, embedder):
        texts = [doc["chunk"] for doc in documents]
        embeddings = embedder.embed_batch(texts)

        ids = []
        metadatas = []

        for i, doc in enumerate(documents):
            ids.append(str(uuid.uuid4()))

            metadatas.append({
                "source": doc["source"],
                "id": doc["id"],
                "chunk_index": i
            })

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query: str, embedder, k=5):
        query_embedding = embedder.embed(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results