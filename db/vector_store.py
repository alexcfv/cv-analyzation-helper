from models.search_results import SearchResultItem
import chromadb
import uuid
from collections import defaultdict

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

    def get_all_grouped_by_source(self) -> dict[str, list[str]]:
        data = self.collection.get(limit=10000)

        grouped: dict[str, list[str]] = defaultdict(list)

        for source, doc in zip(data["metadatas"], data["documents"]):
            grouped[source["source"]].append(doc)

        return grouped

    def search(self, query: str, embedder, k: int = 15) -> list[SearchResultItem]:
        query_embedding = embedder.embed_batch([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        items = []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            items.append(
                SearchResultItem(
                    text=doc,
                    source=meta["source"],
                    distance=dist,
                    chunk_id=meta["id"]
                )
            )

        return items
