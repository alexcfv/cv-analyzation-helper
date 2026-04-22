from embedding.embedder import MistralEmbedder
from db.vector_store import VectorStore
from ingestion.loader import ResumeLoader
from services.ranking import find_best_candidates
from services.explainer import LLMExplainer
from dotenv import load_dotenv
import chromadb
import os

def main():
    # init
    load_dotenv()
    api_key_mistral = os.getenv("MISTRAL_API_KEY")

    embedder = MistralEmbedder(api_key_mistral)
    explainer = LLMExplainer(api_key_mistral)
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

    #top candidate explanation
    top_candidate = ranked[0][0]

    top_chunks = [r.text for r in results if r.source == top_candidate]

    explanation = explainer.explain(query, top_chunks)

    print("Candidate:", top_candidate)
    print("Explanation:\n", explanation)
        
if __name__ == "__main__":
    main()
