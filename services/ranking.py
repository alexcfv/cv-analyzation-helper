from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from models.search_result import SearchResultItem


def group_by_resume(results: List[SearchResultItem]) -> DefaultDict[str, List[SearchResultItem]]:
    grouped: DefaultDict[str, List[SearchResultItem]] = defaultdict(list)

    for r in results:
        grouped[r.source].append(r)

    return grouped


def compute_scores(grouped: Dict[str, List[SearchResultItem]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    for source, chunks in grouped.items():
        best = min(c.distance for c in chunks)
        avg = sum(c.distance for c in chunks) / len(chunks)

        score = 0.7 * best + 0.3 * avg
        scores[source] = score

    return scores


def rank_candidates(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda x: x[1])


def find_best_candidates(results: List[SearchResultItem]) -> List[Tuple[str, float]]:
    grouped = group_by_resume(results)
    scores = compute_scores(grouped)
    return rank_candidates(scores)