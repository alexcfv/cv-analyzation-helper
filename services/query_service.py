from services.ranking import find_best_candidates


class QueryService:
    def __init__(self, embedder, vector_store, explainer, profile_repository, profile_reranker):
        self.embedder = embedder
        self.vector_store = vector_store
        self.explainer = explainer
        self.profile_repository = profile_repository
        self.profile_reranker = profile_reranker

    def search(self, query: str, top_k: int = 3) -> dict:
        results = self.vector_store.search(query, self.embedder, k=10)
        ranked = find_best_candidates(results)

        sources = [s for s, _ in ranked]
        profiles = self.profile_repository.get_by_sources(sources)

        reranked = self.profile_reranker.rerank(query, ranked, profiles)

        candidates = []
        for source, score, explanation in reranked[:top_k]:
            full_explanation = self.explainer.explain(query, source, results)
            candidates.append({
                "source": source,
                "score": score,
                "explanation": full_explanation,
            })

        return {
            "query": query,
            "candidates": candidates,
        }
