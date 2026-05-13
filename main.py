from embedding.embedder import MistralEmbedder
from db.vector_store import VectorStore
from ingestion.loader import ResumeLoader
from services.ranking import find_best_candidates
from services.explainer import LLMExplainer
from db.sqlite.migrations import init_db
from repositories.profile_repo import ProfileRepository
from services.profile_builder import ProfileBuilder
from dotenv import load_dotenv
import chromadb
import os

def main():
    # init
    init_db()
    load_dotenv()
    api_key_mistral = os.getenv("MISTRAL_API_KEY")

    embedder = MistralEmbedder(api_key_mistral)
    explainer = LLMExplainer(api_key_mistral)
    loader = ResumeLoader()

    profile_repository = ProfileRepository()
    profile_builder = ProfileBuilder(api_key_mistral)
    client = chromadb.PersistentClient(path="./chromadb")
    vector_store = VectorStore(client)

    # load docs
    dir_path = input("Enter resumes dir path: ")
    documents = loader.load_folder(dir_path)

    # test
    query = "python backend developer"

    # add to DB if doc is not indexed
    existing = set()
    if vector_store.collection.count() > 0:
        existing = set(
            m["source"] for m in vector_store.collection.get(limit=10000)["metadatas"]
        )

    new_docs = [d for d in documents if d["source"] not in existing]

    if new_docs:
        vector_store.add_documents(new_docs, embedder)
        print(f"Indexed {len(new_docs)} chunks from new files")

    results = vector_store.search(query, embedder)

    ranked = find_best_candidates(results)

    for source, score in ranked:
        print(source, score)

    # top candidate explanation
    top_candidate = ranked[0][0]

    explanation = explainer.explain(query, top_candidate, results)

    print("Candidate:", top_candidate)
    print("Explanation:\n", explanation)

    # add profiles
    all_grouped = vector_store.get_all_grouped_by_source()
    for source, chunks in all_grouped.items():
        if not profile_repository.profile_exists(source):
            profile = profile_builder.build_profile(chunks)
            profile_repository.create_profile(source, profile)
            print(f"Profile saved for {source}")

    print(profile_repository.get_all())

        
if __name__ == "__main__":
    main()
