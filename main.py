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
    dir_path = input("Enter resumes dir path: ")
    documents = loader.load_folder(dir_path)

    # add to DB
    store.add_documents(documents, embedder)

    print("✅ Indexing complete")

    # test
    results = store.search("python backend developer", embedder)
    for result in results:
        print(result.text, result.source)
        
if __name__ == "__main__":
    main()