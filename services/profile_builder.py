from openai import OpenAI
import json


class ProfileBuilder:
    def __init__(self, api_key: str, rate_limiter=None):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            timeout=60
        )
        self.rate_limiter = rate_limiter

    def build_profile(self, chunks: list[str]) -> dict:
        context = "\n".join(chunks)

        prompt = f"""
You are an AI that extracts structured candidate profiles from resume text.

Resume parts:
{context}

Return JSON with:
- summary
- skills (list)
- experience (list)
- education (list)
- projects (list)

Only return valid JSON.
"""

        if self.rate_limiter:
            self.rate_limiter.wait()
        response = self.client.chat.completions.create(
            model="mistral-small-latest",
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Empty response from LLM")

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from LLM: {content}")
