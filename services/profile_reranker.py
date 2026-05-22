from openai import OpenAI
import json


class ProfileReranker:
    def __init__(self, api_key: str, rate_limiter=None):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            timeout=60
        )
        self.rate_limiter = rate_limiter

    def rerank(
        self,
        query: str,
        ranked: list[tuple[str, float]],
        profiles: dict[str, dict],
    ) -> list[tuple[str, float, str]]:
        if not ranked:
            return []

        profiles_text = []
        for source, _ in ranked:
            row = profiles.get(source)
            if not row:
                continue

            profile_raw = row.get("profile", "{}")
            if isinstance(profile_raw, str):
                try:
                    p = json.loads(profile_raw)
                except json.JSONDecodeError:
                    p = {}
            else:
                p = profile_raw

            summary = p.get("summary", "N/A")
            skills = ", ".join(p.get("skills", [])) if isinstance(p.get("skills"), list) else p.get("skills", "N/A")
            experience = "; ".join(
                f"{e.get('role', 'N/A')} at {e.get('company', 'N/A')}"
                for e in (p.get("experience") or [])
            ) or "N/A"

            profiles_text.append(
                f"{source}\n"
                f"  Summary: {summary}\n"
                f"  Skills: {skills}\n"
                f"  Experience: {experience}"
            )

        prompt = f"""You are a strict technical recruiter. Rate each candidate from 1 to 10 based on how well they fit the job.

Job: {query}

Return ONLY valid JSON with source as key and integer score as value. Example:
{{"file1.pdf": 8, "file2.pdf": 3}}

Candidates:
{chr(10).join(f"{i+1}. {t}" for i, t in enumerate(profiles_text))}"""

        if self.rate_limiter:
            self.rate_limiter.wait()

        response = self.client.chat.completions.create(
            model="mistral-small-2603",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content or "{}"
        try:
            llm_scores = json.loads(content)
        except json.JSONDecodeError:
            llm_scores = {}

        results = []
        for source, emb_score in ranked:
            llm_score = llm_scores.get(source, 5)
            try:
                llm_score = float(llm_score)
            except (ValueError, TypeError):
                llm_score = 5.0

            llm_score = max(1, min(10, llm_score))
            final_score = 0.3 * emb_score + 0.7 * (llm_score / 10)
            results.append((source, final_score, f"LLM score: {llm_score:.0f}/10"))

        results.sort(key=lambda x: x[1])
        return results[:3]
