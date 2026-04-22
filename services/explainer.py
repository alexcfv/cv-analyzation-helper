from openai import OpenAI


class LLMExplainer:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1"
        )

    def explain(self, query: str, chunks: list[str]) -> str:
        context = "\n".join(chunks[:3])

        prompt = f"""
You are an AI recruiter.

Job requirements:
{query}

Candidate resume parts:
{context}

Explain why this candidate fits the job.
Use bullet points.
"""

        response = self.client.chat.completions.create(
            model="mistral-small-latest",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content
