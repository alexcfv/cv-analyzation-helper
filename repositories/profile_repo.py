from db.sqlite.connection import get_db
import uuid

class ProfileRepository:
    def add_profiles(self, profiles: list[dict]):
        """
        profiles = [
            {
                "source": "...",
                "profile": "..."
            }
        ]
        """

        rows = []

        for p in profiles:
            rows.append((
                str(uuid.uuid4()),
                p["source"],
                p["profile"]
            ))

        with get_db() as conn:
            conn.execute("""
                INSERT INTO profiles (uuid, source, profile)
                VALUES (?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                profile = excluded.profile
            """, (uuid, source, profile))

    def create_profile(self, profile_source: str, profile: dict):
        profile_uuid = str(uuid.uuid4())

        with get_db() as conn:
            conn.execute(
                "INSERT INTO profiles (uuid, profile_source, profile) VALUES (?, ?, ?)",
                (profile_uuid, profile_source, str(profile))
            )

    def get_all(self):
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM profiles").fetchall()
            return [dict(row) for row in rows]
