import requests
import os
from dotenv import load_dotenv

load_dotenv()

class MistralEmbedder:
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.url = "https://api.mistral.ai/v1/embeddings"

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistral-embed",
                "input": text
            }
        )

        data = response.json()

        return data["data"][0]["embedding"]
    
    def embed_batch(self, texts: list[str], batch_size=32) -> list[list[float]]:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mistral-embed",
                    "input": batch
                }
            )

            if response.status_code != 200:
                raise Exception(f"API Error: {response.text}")

            data = response.json()

            if "data" not in data:
                raise Exception(f"Bad response: {data}")

            embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(embeddings)

        return all_embeddings