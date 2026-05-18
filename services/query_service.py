from services.ranking import find_best_candidates


class QueryService:
    def __init__(self, embedder, vector_store, explainer):
        self.embedder = embedder
        self.vector_store = vector_store
        self.explainer = explainer

    def search(self, query: str, top_k: int = 3) -> dict:
        results = self.vector_store.search(query, self.embedder)
        ranked = find_best_candidates(results)

        top = ranked[:top_k]
        candidates = []
        for source, score in top:
            explanation = self.explainer.explain(query, source, results)
            candidates.append({
                "source": source,
                "score": score,
                "explanation": explanation,
            })

        return {
            "query": query,
            "candidates": candidates,
        }
