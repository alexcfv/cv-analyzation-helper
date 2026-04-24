from openai import OpenAI
import json


class ProfileBuilder:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

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

        response = self.client.chat.completions.create(
            model="mistral-small-lastest",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return json.loads(response.choices[0].message.content)
