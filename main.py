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
    dir_path = input("Endet resumes dir path: ")
    documents = loader.load_folder(dir_path)

    # add to DB
    store.add_documents(documents, embedder)

    print("✅ Indexing complete")

    #Test
    print(store.search("python backend developer", embedder))

if __name__ == "__main__":
    main()