from db.sqlite.connection import get_db

class ProfileRepository:

    def create_profile(self, profile: dict):
        with get_db() as conn:
            conn.execute(
                "INSERT INTO profies (profile) VALUES (?)",
                (profile,)
            )

    def get_all(self):
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM profiles").fetchall()
            return [dict(row) for row in rows]
