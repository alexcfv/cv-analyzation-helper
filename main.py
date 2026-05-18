from embedding.embedder import MistralEmbedder
from db.vector_store import VectorStore
from ingestion.loader import ResumeLoader
from services.explainer import LLMExplainer
from services.index_service import IndexService
from services.query_service import QueryService
from db.sqlite.migrations import init_db
from repositories.profile_repo import ProfileRepository
from services.profile_builder import ProfileBuilder
from dotenv import load_dotenv
import chromadb
import os


def main():
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

    index_service = IndexService(embedder, loader, vector_store, profile_builder, profile_repository)
    query_service = QueryService(embedder, vector_store, explainer)

    dir_path = input("Enter resumes dir path: ")
    result = index_service.index_folder(dir_path)
    print(f"Indexed {result['new_chunks']} chunks, {len(result['new_profiles'])} new profiles")

    query = input("Enter search query: ")
    search_result = query_service.search(query)

    for c in search_result["candidates"]:
        print(f"\n--- {c['source']} (score: {c['score']:.4f}) ---")
        print(c["explanation"])


if __name__ == "__main__":
    main()
