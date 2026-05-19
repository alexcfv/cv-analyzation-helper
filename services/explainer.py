from openai import OpenAI
from models.search_results import SearchResultItem

class LLMExplainer:
    def __init__(self, api_key: str, rate_limiter=None):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1"
        )
        self.rate_limiter = rate_limiter

    def explain(self, query: str, candidate: str, vectors: list[SearchResultItem]) -> list[str]:
        vectors.sort(key=lambda x: x.distance)

        candidate_chunks = [r.text for r in vectors if r.source == candidate]

        context = "\n".join(candidate_chunks[:3])

        prompt = f"""
You are an AI recruiter.

Job requirements:
{query}

Candidate resume parts:
{context}

Explain why this candidate fits the job.
Use bullet points.
"""

        if self.rate_limiter:
            self.rate_limiter.wait()
        response = self.client.chat.completions.create(
            model="mistral-small-latest",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content
