import requests
import os

class MistralEmbedder:
    def __init__(self, api_key, rate_limiter=None):
        self.api_key = api_key
        self.url = "https://api.mistral.ai/v1/embeddings"
        self.rate_limiter = rate_limiter

    def embed(self, text: str) -> list[float]:
        if self.rate_limiter:
            self.rate_limiter.wait()
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

            if self.rate_limiter:
                self.rate_limiter.wait()
            response = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mistral-embed",
                    "input": batch
                },
                timeout=60
            )

            if response.status_code != 200:
                raise Exception(f"API Error: {response.text}")

            data = response.json()

            if "data" not in data:
                raise Exception(f"Bad response: {data}")

            embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(embeddings)

        return all_embeddings
