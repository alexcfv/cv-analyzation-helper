from collections import defaultdict
from models.search_results import SearchResultItem


def group_by_resume(results: list[SearchResultItem]):
    grouped = defaultdict(list)

    for r in results:
        grouped[r.source].append(r)

    return grouped


def compute_scores(grouped):
    scores = {}

    for source, chunks in grouped.items():
        best = min(c.distance for c in chunks)
        avg = sum(c.distance for c in chunks) / len(chunks)

        score = 0.7 * best + 0.3 * avg
        scores[source] = score

    return scores


def rank_candidates(scores):
    return sorted(scores.items(), key=lambda x: x[1])


def find_best_candidates(results: list[SearchResultItem]):
    grouped = group_by_resume(results)
    scores = compute_scores(grouped)
    return rank_candidates(scores)