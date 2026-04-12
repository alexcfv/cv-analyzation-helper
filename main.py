from embedding.embedder import MistralEmbedder
from db.vector_store import VectorStore
from ingestion.loader import ResumeLoader
import chromadb

def main():
    # init
    embedder = MistralEmbedder()
    loader = ResumeLoader()

    client = chromadb.Client()
    store = VectorStore(client)

    # load docs
    documents = loader.load_folder("/home/rzhavii/Documents/resumes")

    # add to DB
    store.add_documents(documents, embedder)

    print("✅ Indexing complete")

if __name__ == "__main__":
    main()