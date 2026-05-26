from openai import OpenAI
from models.search_results import SearchResultItem

class LLMExplainer:
    def __init__(self, api_key: str, model="mistral-small-2603", timeout=60, rate_limiter=None):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            timeout=timeout
        )
        self.rate_limiter = rate_limiter

    def explain(self, query: str, candidate: str, vectors: list[SearchResultItem]) -> list[str]:
        vectors.sort(key=lambda x: x.distance)

        candidate_chunks = [r.text for r in vectors if r.source == candidate]

        context = "\n".join(candidate_chunks[:3])

        prompt = f"""
You are a technical recruiter. Given job requirements and candidate resume parts, respond in under 50 words.

Summarize:
- Key skills matched
- Company they worked at
- Relevant tasks/projects they handled

Be concise. No fluff. No bullet points. One short paragraph.

Job requirements:
{query}

Candidate resume parts:
{context}
"""

        if self.rate_limiter:
            self.rate_limiter.wait()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content
