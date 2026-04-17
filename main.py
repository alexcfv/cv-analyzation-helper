from embedding.embedder import MistralEmbedder
from db.vector_store import VectorStore
from ingestion.loader import ResumeLoader
from services.ranking import find_best_candidates
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
    query = "python backend developer"

    results = store.search(query, embedder)

    ranked = find_best_candidates(results)

    for source, score in ranked:
        print(source, score)
        
if __name__ == "__main__":
    main()