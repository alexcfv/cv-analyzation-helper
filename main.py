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
    client = chromadb.Client()
    vectore_store = VectorStore(client)

    # load docs
    dir_path = input("Enter resumes dir path: ")
    documents = loader.load_folder(dir_path)

    # add to DB
    vectore_store.add_documents(documents, embedder)

    print("✅ Indexing complete")

    # test
    query = "python backend developer"

    results = vectore_store.search(query, embedder)

    ranked = find_best_candidates(results)

    for source, score in ranked:
        print(source, score)

    #top candidate explanation
    top_candidate = ranked[0][0]

    top_candidate_chunks = [r.text for r in results if r.source == top_candidate]

    explanation = explainer.explain(query, top_candidate_chunks)

    print("Candidate:", top_candidate)
    print("Explanation:\n", explanation)

    #profile builder
    top_candidate_profile = profile_builder.build_profile(top_candidate_chunks)

    #insert profile into db
    profile_repository.create_profile(top_candidate, top_candidate_profile)
    print(profile_repository.get_all())

        
if __name__ == "__main__":
    main()
