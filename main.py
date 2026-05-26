from embedding.embedder import MistralEmbedder
from db.vector_store import VectorStore
from ingestion.loader import ResumeLoader
from services.explainer import LLMExplainer
from services.index_service import IndexService
from services.query_service import QueryService
from db.sqlite.migrations import init_db
from repositories.profile_repo import ProfileRepository
from services.profile_builder import ProfileBuilder
from services.profile_reranker import ProfileReranker
from services.rate_limiter import RateLimiter
from config import load_config
import chromadb


def main():
    cfg = load_config()
    init_db()
    api_key_mistral = cfg["api"]["mistral_key"]

    rate_limiter = RateLimiter(min_interval=cfg["rate_limiter"]["min_interval"])

    embedder = MistralEmbedder(api_key_mistral, model=cfg["embedder"]["model"], timeout=cfg["embedder"]["timeout"], rate_limiter=rate_limiter)
    explainer = LLMExplainer(api_key_mistral, model=cfg["explainer"]["model"], timeout=cfg["explainer"]["timeout"], rate_limiter=rate_limiter)
    loader = ResumeLoader()
    profile_repository = ProfileRepository()
    profile_builder = ProfileBuilder(api_key_mistral, model=cfg["profile_builder"]["model"], timeout=cfg["profile_builder"]["timeout"], rate_limiter=rate_limiter)
    profile_reranker = ProfileReranker(api_key_mistral, model=cfg["reranker"]["model"], timeout=cfg["reranker"]["timeout"], rate_limiter=rate_limiter)
    client = chromadb.PersistentClient(path="./chromadb")
    vector_store = VectorStore(client)

    index_service = IndexService(embedder, loader, vector_store, profile_builder, profile_repository)
    query_service = QueryService(embedder, vector_store, explainer, profile_repository, profile_reranker)

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
