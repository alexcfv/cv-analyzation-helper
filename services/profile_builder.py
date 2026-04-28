from openai import OpenAI
import json


class ProfileBuilder:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1"
        )

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
            print("Invalid JSON:", content)
        except Exception as e:
            raise e
