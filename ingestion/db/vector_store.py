import chromadb

class VectorStore:
    def __init__(self, path: str):
        self.client = chromadb.Client(
            chromadb.config.Settings(
                persist_directory=path
            )
        )

        self.collection = self.client.get_or_create_collection(
            name="resumes"
        )
    
    def add_documents(self, documents, embedder):
        ids = []
        texts = []
        metadatas = []
        embeddings = []

        for i, doc in enumerate(documents):
            chunk_id = f"{doc['source']}_{i}"

            ids.append(chunk_id)
            texts.append(doc["chunk"])
            metadatas.append({
                "source": doc["source"],
                "id": doc["id"]
            })

            embedding = embedder(doc["chunk"])
            embeddings.append(embedding)

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def search(self, query_embedding, k=5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results