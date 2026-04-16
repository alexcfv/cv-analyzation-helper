from dataclasses import dataclass

@dataclass
class SearchResultItem:
    text: str
    source: str
    distance: float
    chunk_id: str